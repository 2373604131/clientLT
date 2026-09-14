"""Exercise the actual CAPT client loop with small CPU models, without CLIP/data."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.run_capt_global_start import build_command, ensure_schedule, main, parse_args

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'federated_main.py').read_text(encoding='utf-8-sig'))


def actual_loop():
    matches = [n for n in ast.walk(TREE) if isinstance(n, ast.For)
               and isinstance(n.target, ast.Name) and n.target.id == 'idx'
               and isinstance(n.iter, ast.Name) and n.iter.id == 'idxs_users'
               and 'local_coupling_params[idx]' in ast.unparse(n)]
    assert len(matches) == 1
    return compile(ast.Module(body=matches, type_ignores=[]), '<CAPT client loop>', 'exec')


class ToyTrainer:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.prompt_learner = torch.nn.Linear(1, 1)
        self.model.coupling_function = torch.nn.Linear(1, 1)
        self.model.register_buffer('server_buffer', torch.tensor([0.]))
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.starts = []
        self.reset_calls = 0

    def reset_optimizer_and_scheduler(self):
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.reset_calls += 1

    def train(self, idx, **kwargs):
        self.starts.append(copy.deepcopy(self.model.state_dict()))
        self.optim.zero_grad()
        sum(p.sum() * (idx + 1) for p in self.model.parameters()).backward()
        self.optim.step()
        self.model.server_buffer.add_(idx + 1)


def state_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def context(reset, reset_optimizer=False):
    trainer = ToyTrainer()
    server = {k: torch.full_like(v, 3.) for k, v in trainer.model.state_dict().items()}
    return dict(copy=copy, local_trainer=trainer, global_weights=server,
                args=SimpleNamespace(capt_matched_v2=reset_optimizer),
                capt_reset_global_before_client=reset, idxs_users=[0, 1], epoch=0,
                local_weights=[None, None], local_coupling_params=[None, None])


def test_each_client_restores_all_global_state_including_coupling_and_buffers():
    ns = context(True)
    trainer = ns['local_trainer']
    opt = trainer.optim
    server_before = copy.deepcopy(ns['global_weights'])
    exec(actual_loop(), ns)
    assert all(state_equal(start, server_before) for start in trainer.starts)
    assert state_equal(ns['global_weights'], server_before)
    assert trainer.optim is opt  # the new flag does not silently reset momentum
    assert trainer.reset_calls == 0
    assert not state_equal(ns['local_weights'][0], ns['local_weights'][1])


def test_skipped_aggregation_reuses_last_server_then_new_broadcast_is_used():
    ns = context(True)
    code = actual_loop()
    exec(code, ns)
    ns['epoch'] = 1  # no server aggregation occurred
    exec(code, ns)
    assert all(state_equal(s, ns['global_weights']) for s in ns['local_trainer'].starts)
    ns['global_weights'] = {k: torch.full_like(v, 7.) for k, v in ns['global_weights'].items()}
    ns['epoch'] = 2
    exec(code, ns)
    assert all(state_equal(s, ns['global_weights']) for s in ns['local_trainer'].starts[-2:])


def test_disabled_flag_preserves_serial_training():
    ns = context(False)
    exec(actual_loop(), ns)
    assert state_equal(ns['local_trainer'].starts[1], ns['local_weights'][0])
    assert not state_equal(ns['local_trainer'].starts[1], ns['global_weights'])


def test_existing_optimizer_reset_can_be_enabled_independently():
    ns = context(True, reset_optimizer=True)
    exec(actual_loop(), ns)
    assert ns['local_trainer'].reset_calls == 2
    assert all(state_equal(s, ns['global_weights']) for s in ns['local_trainer'].starts)


def test_real_cli_flag_is_opt_in_and_validation_rejects_other_models():
    flag = '--capt_reset_global_before_client'
    calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag]
    assert len(calls) == 1
    parser = argparse.ArgumentParser()
    node = ast.Expr(value=calls[0])
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<CAPT CLI>', 'exec'),
         {'parser': parser, 'str2bool': lambda x: x.lower() == 'true'})
    assert parser.parse_args([]).capt_reset_global_before_client is False
    assert parser.parse_args([flag, 'True']).capt_reset_global_before_client is True
    checks = [n for n in ast.walk(TREE) if isinstance(n, ast.If)
              and 'capt_reset_global_before_client requires' in ast.unparse(n)
              and isinstance(n.test, ast.BoolOp)
              and isinstance(n.test.values[0], ast.Name)
              and n.test.values[0].id == 'capt_reset_global_before_client']
    assert len(checks) == 1
    code = compile(ast.Module(body=checks, type_ignores=[]), '<CAPT validation>', 'exec')
    for trainer, model in [('CAPT', 'fedavg'), ('PromptFL', 'cluster')]:
        with pytest.raises(ValueError, match='requires model=cluster'):
            exec(code, {'capt_reset_global_before_client': True,
                        'args': SimpleNamespace(trainer=trainer, model=model)})


def test_launcher_pairs_partitions_and_changes_only_requested_defaults(tmp_path):
    args = parse_args(['--output-root', str(tmp_path / 'runs'), '--seeds', '42'])
    commands = [build_command(args, 42, p) for p in args.partitions]
    opts = [dict(zip(c[3::2], c[4::2])) for c in commands]
    assert {o['--partition'] for o in opts} == {'client-longtail', 'noniid-labeldir-fine'}
    for o in opts:
        assert o['--model'] == 'cluster' and o['--trainer'] == 'CAPT'
        assert o['--capt_reset_global_before_client'] == 'True'
        assert o['--capt_fixed_global_agg_freq'] == '0'
        assert o['--capt_matched_v2'] == 'False'
    assert opts[0]['--client_schedule_file'] == opts[1]['--client_schedule_file']
    assert ensure_schedule(args, 42) == ensure_schedule(args, 42)


def test_launcher_dry_run_writes_nothing_and_existing_results_are_protected(tmp_path, capsys):
    out = tmp_path / 'runs'
    main(['--output-root', str(out), '--seeds', '42', '--dry-run'])
    assert not out.exists()
    assert '--capt_reset_global_before_client True' in capsys.readouterr().out
    run = out / 'seed42' / 'client-longtail'
    run.mkdir(parents=True)
    (run / 'result.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        main(['--output-root', str(out), '--seeds', '42'])
    assert (run / 'result.txt').read_text() == 'keep'
