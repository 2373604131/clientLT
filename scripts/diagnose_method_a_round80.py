"""Inspect logs or replay ONE saved margin-A correction. Never run local training.

History can be reconstructed by forward-only observations of saved submitted
states: round k's committed model is round k+1 normal_B's anchor. No trajectory
is regenerated. All output goes outside the source run.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RUN_SUFFIX = Path('new_a_joint/runs/seed42/method_a_margin')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker_arguments(recorded_command):
    """Discard Python and its flags as well as the recorded worker script."""
    worker = next(i for i, value in enumerate(recorded_command)
                  if Path(value).name == 'train_cliplora_joint_aggregation.py')
    return list(recorded_command[worker+1:])


def event_path(run, rnd, factor):
    return run/'events'/f'r{rnd:03d}_c000_main_{factor}'/'state.pt'


def inspect_run(run, rnd=80):
    import numpy as np
    config = read_json(run/'sfra_config.json')
    job = read_json(run/'joint_job.json')
    if (job['arm'] != 'method_a_margin' or job['aggregation'].get('rule', 'joint') != 'joint'
            or 'b_transfer' in config or config['variant'] != 'margin-keep-v1'):
        raise ValueError('Require the original joint + new A, with new B disabled')
    selected = next(r for r in rows(run/'sfra_rounds.csv') if int(r['round']) == rnd)
    event = read_json(event_path(run, rnd, 'refresh_A').with_name('event.json'))
    radius, z_norm = float(selected['radius']), float(selected['correction_z_norm'])
    ideal = radius*z_norm
    required = [run/'checkpoints/base_model.pt', event_path(run, rnd, 'refresh_A')]
    history_states = [event_path(run, k, 'normal_B') for k in range(1, rnd+1)]
    command = read_json(run/'command.json')
    original_run = command[command.index('--output-dir')+1]
    result = dict(status='logged_evidence_only', full_gradient_diagnosis_completed=False,
        source_run=str(run), original_server_run=original_run, round=rnd,
        valid_history_count=int(selected['valid_history_count']),
        violated_count=int(selected['ordinary_deficit_count']),
        violated_fraction=(int(selected['ordinary_deficit_count'])/int(selected['valid_history_count'])
                           if int(selected['valid_history_count']) else None),
        history_last_used=int(selected['history_last_used']),
        L_keep=float(selected['ordinary_keep']), L_keep_after=float(selected['final_keep']),
        C_label=float(selected['ordinary_class']), C_label_after=float(selected['final_class']),
        R_t=radius, correction_z_norm=z_norm, ideal_delta_A_norm_from_R_z=ideal,
        ordinary_delta_norm=float(event['delta_a_norm']),
        ideal_update_ratio_from_logs=ideal/float(event['delta_a_norm']),
        ordinary_effective_weight_norm=float(event['effective_delta_norm']),
        repair_effective_weight_norm=None, delta_A_norm_actual_tensor_difference=None,
        grad_keep_norm=None, grad_learn_norm=None, grad_cosine=None, grad_total_norm=None,
        first_step_predicted_update=None, first_step_actual_update=None, projection_triggered=None,
        witness_correct_to_wrong=None, witness_wrong_to_correct=None,
        lambda_keep=config['keep_weight'], lambda_learn=config['learn_weight'],
        step_size=config['correction_step_size'], correction_steps=int(selected['correction_steps']),
        first_step_effective_A_learning_rate=config['correction_step_size']*radius**2,
        violated_count_after=int(selected['final_deficit_count']),
        logged_nonzero_correction=selected['nonzero_correction']=='True',
        group_classification={g: dict(before=float(selected['ordinary_CE_'+g]),
                                     after=float(selected['final_CE_'+g]),
                                     change=float(selected['final_CE_'+g])-float(selected['ordinary_CE_'+g]))
                              for g in ('H', 'M', 'T')},
        artifacts=dict(required_proposal_and_base=[str(p) for p in required],
            missing_proposal_and_base=[str(p) for p in required if not p.is_file()],
            saved_history_states_present=sum(p.is_file() for p in history_states),
            saved_history_states_required=rnd,
            explanation='pack_results excludes .pt/checkpoints; sfra_last.pt overwrites prior histories'),
        interpretation=dict(protection_not_triggered='ruled out by nonzero keep loss and recorded deficits'
                            if int(selected['ordinary_deficit_count']) and float(selected['ordinary_keep']) > 0 else 'possible',
            parameterization='unprojected first step is -eta*R_t^2*grad_A; consistent with weak correction',
            final_cause='unresolved without component gradients and actual saved tensors'))
    for name in ('L_keep', 'C_label'):
        result[name+'_change'] = result[name+'_after']-result[name]
    # Official test transitions are explicitly separate from training witnesses.
    old = run/'stage_predictions/ordinary_A'/f'r{rnd:03d}.npz'
    new = run/'predictions'/f'r{rnd:03d}.npz'
    if old.is_file() and new.is_file():
        with np.load(old) as a, np.load(new) as b:
            if not all(np.array_equal(a[k], b[k]) for k in ('sample_id', 'class_id')):
                raise ValueError('Stage prediction identities differ')
            result['official_test_only'] = dict(count=len(a['correct']),
                correct_to_wrong=int((a['correct'] & ~b['correct']).sum()),
                wrong_to_correct=int((~a['correct'] & b['correct']).sum()),
                changed_predictions=int((a['prediction'] != b['prediction']).sum()))
    return result


def saved_event(run, rnd, factor, keys):
    import torch
    from utils.cliplora_a_refresh import state_hash
    path = event_path(run, rnd, factor)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = read_json(path.with_name('event.json'))
    if payload['round'] != rnd:
        raise ValueError('Saved event round mismatch')
    for field, digest in (('anchor_lora_state', 'before_lora_sha256'),
                          ('actual_after_lora_state', 'after_lora_sha256')):
        if state_hash(payload[field], keys) != metadata[digest]:
            raise ValueError('Saved tensor hash mismatch: '+str(path))
    return payload


def verify_initial_numerics(runtime, source, state, observed):
    """Reproduce the saved frozen-prefix convolution before restoring history.

    A channels-last convolution reproduces the original four-image witness
    prefix. The legacy run did not record cuDNN's chosen engine, and current
    autotuning can choose another FP32/TF32 path. Only a bitwise match against
    unchanged original history is accepted. No final-round history values are
    used as protection targets.
    """
    import torch
    saved = torch.load(source/'checkpoints/sfra_last.pt', map_location='cpu', weights_only=False)
    history = saved['functional_history']
    mask = history['valid'] & (history['registered_round']==0)
    if not mask.any():
        raise ValueError('No unchanged original history values for numerical replay verification')
    trials = []

    def check(label, obs):
        error = obs['sample_margins'].amin(1)[mask]-history['H'][mask]
        row = dict(mode=label, count=int(mask.sum()), bitwise_equal_count=int((error==0).sum()),
                   max_abs_error=float(error.abs().max()))
        trials.append(row)
        write_json(runtime.root.parent/'initial_numerics.json', trials)
        print('INITIAL_NUMERICS', json.dumps(row), flush=True)
        return bool((error==0).all())

    if check('original_runtime_initialization', observed):
        runtime.initial_numerics = trials
        return observed
    manifest = runtime.margin_feedback.manifest
    tokens = runtime.bank.tokens
    wrong = mask & (observed['sample_margins'].amin(1)!=history['H'])
    affected_sizes = sorted({len(tokens[manifest[i]['token_id']]['local_positions'])
                             for i in wrong.nonzero().flatten().tolist()})
    cache = runtime.bank._fast_feedback
    convolution = runtime.bank.core.image_encoder.conv1
    original_forward = convolution.forward

    def original_numerical_path(images):
        if len(images) not in affected_sizes:
            return original_forward(images)
        return torch.nn.functional.conv2d(images.contiguous(memory_format=torch.channels_last),
            convolution.weight.contiguous(memory_format=torch.channels_last), convolution.bias,
            convolution.stride, convolution.padding, convolution.dilation, convolution.groups).contiguous()

    try:
        convolution.forward = original_numerical_path
        for key in list(cache.features):
            if len(tokens[key[0]]['local_positions']) in affected_sizes:
                value = cache.features.pop(key)
                size = value.numel()*value.element_size()
                if value.is_cuda:
                    cache.device_cache_bytes -= size
                else:
                    cache.cpu_cache_bytes -= size
        observed = runtime.observe_margin(state, 0, 'verify_original_convolution_algorithm')
        if not check(dict(prefix_batch_sizes=affected_sizes, convolution='channels_last_cudnn'), observed):
            raise ValueError('Frozen-prefix numerics differ from original history; diagnosis stopped')
    finally:
        convolution.forward = original_forward
    runtime.initial_numerics = trials
    return observed


def restore_history(runtime, source, rnd, history_checkpoint=None):
    import torch
    from utils.method_a_margin_math import SampleMarginHistory
    if history_checkpoint:
        payload = torch.load(history_checkpoint, map_location='cpu', weights_only=False)
        if payload['completed_round'] != rnd-1 or payload['sfra_config'] != runtime.sfra_config:
            raise ValueError('History checkpoint must match this run/configuration and round-1')
        history = SampleMarginHistory.restore(payload['functional_history'], runtime.margin_feedback.manifest)
        if history.last_committed_round != rnd-1:
            raise ValueError('History checkpoint round mismatch')
        # Bind a checkpoint to this trajectory, not merely the same algorithm.
        from utils.cliplora_a_refresh import state_hash
        checkpoint_state = dict(runtime.base_state, **payload['state_dict_overrides'])
        anchor = saved_event(source, rnd, 'normal_B', runtime.keys)['anchor_lora_state']
        if state_hash(checkpoint_state, runtime.keys) != state_hash(anchor, runtime.keys):
            raise ValueError('History checkpoint belongs to a different model trajectory')
        return history, 'exact_checkpoint'
    logged = {int(r['round']): r for r in rows(source/'sfra_rounds.csv')}
    history = None
    audit_dir = runtime.root.parent/'history_reconstruction'
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for committed in range(rnd):
        payload = saved_event(source, committed+1, 'normal_B', runtime.keys)
        state = dict(runtime.base_state, **payload['anchor_lora_state'])
        obs = runtime.observe_margin(state, committed, 'history_reconstruction_forward_only')
        if committed == 0 and hasattr(runtime, 'initial_numerics'):
            obs = verify_initial_numerics(runtime, source, state, obs)
        torch.save(obs['sample_margins'], audit_dir/f'margins_r{committed:03d}.pt')
        if committed == 0:
            history = SampleMarginHistory(obs['sample_margins'], runtime.margin_feedback.manifest)
            if int(history.valid.sum()) != int(logged[1]['valid_history_count']):
                raise ValueError('Initial reconstructed history differs from the original log')
        else:
            previous = history.state_dict()
            changed = history.commit(obs['sample_margins'], committed)
            audit_rows.append(dict(round=committed, valid_actual=int(history.valid.sum()),
                valid_expected=int(logged[committed]['history_count_after']), registered_actual=int(changed.sum()),
                registered_expected=int(logged[committed]['history_registered'])))
            write_json(audit_dir/'counts.json', audit_rows)
            if (int(history.valid.sum()) != int(logged[committed]['history_count_after'])
                    or int(changed.sum()) != int(logged[committed]['history_registered'])):
                torch.save(dict(previous=previous, current=history.state_dict(), margins=obs['sample_margins'],
                                counts=audit_rows[-1]), audit_dir/'mismatch.pt')
                raise ValueError(f'Reconstructed history counts differ at round {committed}: {audit_rows[-1]}; '
                                 'detailed tensors saved; do not substitute final history')
    return history, 'forward_only_reconstruction_from_saved_submitted_models'


def replay(args, evidence):
    source, out, rnd = args.run, args.output, args.round
    required = [source/'checkpoints/base_model.pt', event_path(source, rnd, 'refresh_A')]
    required += ([args.history_checkpoint, event_path(source, rnd, 'normal_B')] if args.history_checkpoint else
                 [event_path(source, k, 'normal_B') for k in range(1, rnd+1)])
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} saved tensor files. First: {missing[0]}. No training started.')
    job = read_json(source/'joint_job.json')
    from tools.client_aggregation.numerics import policy_for, prepare_environment, apply_policy
    prepare_environment(policy_for(job))
    apply_policy(policy_for(job))
    import torch
    import runpy
    from tools.client_aggregation.protocol import code_hashes
    from tools.client_aggregation.runtime import runtime_class
    from utils import cliplora_sfra
    from utils.method_a_margin_math import make_targets
    from utils.method_a_update_diagnostic import diagnose_prepared, difference, norm
    from utils.sfra_math import flatten
    current = code_hashes(REPO)
    changed = [k for k, v in job['code_sha256'].items() if current.get(k) != v]
    allowed = {'utils/cliplora_method_a_margin.py', 'utils/method_a_margin_feedback.py'}
    if set(changed)-allowed:
        raise ValueError('Non-diagnostic source changes: '+str(sorted(set(changed)-allowed)))
    torch.set_num_threads(1)
    work = out/'runtime'
    if work.exists():
        raise ValueError('Use a fresh output directory for replay')
    shutil.copytree(source/'protocol', work/'protocol')
    original_runtime = cliplora_sfra.SFRARuntime

    class DiagnosticRuntime(runtime_class(job)):
        def __init__(self, *a, **kw):
            # Preserve production initialization and CUDA allocation order too.
            # The dormant client pool never executes a training phase.
            super().__init__(*a, **kw)
            self.initial_numerics = []

        def train_phase(self, *a, **kw):
            raise RuntimeError('Local training is disabled in the diagnostic worker')

        def checkpoint(self, *a, **kw):
            raise RuntimeError('Training checkpoints are disabled in the diagnostic worker')

        def publish(self, *a, **kw):
            raise RuntimeError('Training publication is disabled in the diagnostic worker')

        def run(self, initial_state):
            self.base_state = torch.load(source/'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
            from utils.cliplora_a_refresh import state_hash
            frozen_keys = sorted(set(self.base_state)-set(self.keys))
            if state_hash(self.base_state, frozen_keys) != self.audit.meta['frozen_model_sha256']:
                raise ValueError('Saved base model differs from reconstructed frozen backbone')
            if self.margin_feedback.manifest != read_json(source/'margin_witness_manifest.json'):
                raise ValueError('Witness identities differ from the original run')
            self.history, history_source = restore_history(self, source, rnd, args.history_checkpoint)
            torch.save(self.history.state_dict(), out/f'history_before_r{rnd:03d}.pt')
            event = saved_event(source, rnd, 'refresh_A', self.keys)
            middle = dict(self.base_state, **event['anchor_lora_state'])
            ordinary = dict(self.base_state, **event['actual_after_lora_state'])
            radius = float(torch.stack([flatten(d, self.a_keys).square().sum()
                                       for d in event['local_factor_deltas']]).mean().sqrt())
            if abs(radius-evidence['R_t']) > 1e-8:
                raise ValueError('Saved client proposal radius differs from original log')
            radius = evidence['R_t']  # Preserve the original GPU reduction's exact recorded scalar.
            targets = make_targets(self.history, self.margin_feedback.base_mass,
                                   self.sfra_config['epsilon'], self.sfra_config['margin_scale'])
            before = self.observe_margin(ordinary, rnd, 'verify_restored_ordinary', targets=targets)
            checks = dict(valid_history_count=int(self.history.valid.sum())==evidence['valid_history_count'],
                violated_count=before['deficit_count']==evidence['violated_count'],
                keep_loss=abs(before['keep_loss']-evidence['L_keep']) <= 1e-6+1e-5*abs(evidence['L_keep']),
                class_loss=abs(before['class_loss']-evidence['C_label']) <= 1e-6+1e-5*abs(evidence['C_label']))
            write_json(out/'restoration_checks.json', dict(checks=checks, history_source=history_source,
                       L_keep=before['keep_loss'], C_label=before['class_loss']))
            if not all(checks.values()):
                raise ValueError('Restored training state/history does not reproduce original round measurements')
            submitted, _, result = diagnose_prepared(self, middle, ordinary, radius, rnd, out)
            comparison = event_path(source, rnd+1, 'normal_B')
            if comparison.is_file():
                recorded = saved_event(source, rnd+1, 'normal_B', self.keys)['anchor_lora_state']
                result['replayed_vs_original_submitted_A_error_norm'] = norm(difference(submitted, recorded, self.a_keys))
            result.update(status='single_saved_round_replayed', history_source=history_source,
                          local_training_steps=0, source_run=str(source), original_code_changes=changed,
                          device=str(self.trainer.device), torch_version=str(torch.__version__),
                          initial_numerics=self.initial_numerics)
            write_json(out/'diagnosis.json', result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    command = worker_arguments(read_json(source/'command.json'))
    def replace(flag, value):
        command[command.index(flag)+1] = str(value)
    index = command.index('--joint-spec')
    del command[index:index+2]
    replace('--output-dir', work)
    replace('--client_schedule_file', work/'protocol/full_schedule.json')
    replace('--lac_partition_manifest', work/'protocol/partition_source.csv')
    replace('--sfra_resume', '')
    if args.data_root:
        replace('--root', args.data_root)
    write_json(out/'replay_receipt.json', dict(source_run=str(source), command=command,
               allowed_diagnostic_code_changes=changed, current_code_sha256=current,
               registered_code_sha256=job['code_sha256'], local_training_enabled=False))
    argv = sys.argv
    try:
        cliplora_sfra.SFRARuntime = DiagnosticRuntime
        sys.argv = [str(REPO/'federated_main.py'), *command]
        runpy.run_path(str(REPO/'federated_main.py'), run_name='__main__')
    finally:
        cliplora_sfra.SFRARuntime, sys.argv = original_runtime, argv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    server = REPO/'output/cifar100_LT/ab_factorial_joint_v1_parallel8'/RUN_SUFFIX
    copied = REPO/'output/ab_factorial_joint_v1_parallel8'/RUN_SUFFIX
    p.add_argument('--run', type=Path, default=server if server.is_dir() else copied)
    p.add_argument('--round', type=int, choices=(80,), default=80)
    p.add_argument('--output', type=Path, default=REPO/'output/method_a_round80_diagnosis')
    p.add_argument('--replay', action='store_true', help='Run the saved correction only; requires original .pt files')
    p.add_argument('--history-checkpoint', type=Path, help='Optional original round79 checkpoint; otherwise restore from saved models')
    p.add_argument('--data-root', type=Path)
    args = p.parse_args(argv)
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.output == args.run or args.run in args.output.parents or args.output in args.run.parents:
        p.error('Output must be separate from the original run')
    try:
        evidence = inspect_run(args.run, args.round)
        write_json(args.output/'logged_evidence.json', evidence)
        print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
        if args.replay:
            replay(args, evidence)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
