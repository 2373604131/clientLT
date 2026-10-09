"""CPU checks for the seed42 controls; no CIFAR/CLIP training or downloads."""
import copy
import hashlib
import shutil
import tempfile
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_method_a_simple_controls as launcher
from scripts.run_ab_validation import write_json
from tools.sfra import simple_controls as protocol
from tools.sfra.simple_controls_summary import audit_control_logs, pack, read_result, sample_metrics, summarize
from utils.cliplora_a_refresh import aggregate_refresh_deltas
from utils.cliplora_sfra import SFRARuntime
from utils.lora_aggregation import aggregate_lora_state
from utils.method_a_simple_controls import MethodASimpleControlsRuntime as Runtime
from utils.sfra_math import FunctionalHistory
from test_sfra_execution import visual_bank

REPO = Path(__file__).resolve().parents[1]


class TestSimpleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_partition_uses_all_training_examples_and_rejects_duplicates(self):
        rows = [dict(client_id=j, class_id=c, local_position=i, raw_sample_id=10*j+i)
                for j, labels in enumerate(([0, 0, 1], [1, 2], [0, 2, 2]))
                for i, c in enumerate(labels)]
        counts = protocol.partition_counts(rows, [2])
        self.assertEqual(counts, dict(sizes=[3, 2, 3], tail_counts=[0, 1, 2],
            class_counts=[3, 2, 3], coverage=[2, 2, 2], tail_ids=[2]))
        with self.assertRaises(ValueError):
            protocol.partition_counts(rows+[rows[0]], [2])
        broken = copy.deepcopy(rows)
        broken[0]['local_position'] = 9
        with self.assertRaises(ValueError):
            protocol.partition_counts(broken, [2])

    def test_tailrw_closed_form_zero_and_invalid_strengths(self):
        sizes, tails = [80, 20], [0, 10]
        self.assertEqual(protocol.tail_weights(sizes, tails, 0), [.8, .2])
        self.assertEqual(protocol.tail_weights(sizes, tails, 4), [4/7, 3/7])
        for bad in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                protocol.tail_weights(sizes, tails, bad)

    def test_probe_replay_recovers_recorded_bytes_in_both_newline_directions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'probe_manifest.csv'
            lf = b'class_id,slot,raw_train_index,excluded_from_federated_lt_pool\n80,0,24418,1\n'
            crlf = lf.replace(b'\n', b'\r\n')
            for original, transported, conversion in ((crlf, lf, 'CRLF'), (lf, crlf, 'LF'), (crlf, crlf, 'unchanged')):
                path.write_bytes(transported)
                expected = hashlib.sha256(original).hexdigest()
                payload, receipt = protocol.probe_manifest_replay(path, expected)
                self.assertEqual(payload, original)
                self.assertEqual(receipt['newline_conversion'], conversion)
                self.assertEqual(receipt['replayed_sha256'], expected)
                self.assertEqual(path.read_bytes(), transported)  # Reference is never mutated.

    def test_probe_replay_refuses_real_sample_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'probe_manifest.csv'
            original = b'class_id,slot,raw_train_index\r\n80,0,24418\r\n'
            altered = original.replace(b'24418', b'24419').replace(b'\r\n', b'\n')
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, 'cannot be repaired'):
                protocol.probe_manifest_replay(path, hashlib.sha256(original).hexdigest())
            self.assertEqual(path.read_bytes(), altered)

    def test_all_four_launches_replay_linux_transport_without_changing_reference(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            reference = root/'reference'
            shutil.copytree(REPO/'references/full10_clientlt', reference)
            manifest = reference/'protocol/probe_manifest.csv'
            original = manifest.read_bytes()
            expected = protocol.load_json(reference/'bridge_metadata.json')['probe_manifest_sha256']
            self.assertEqual(hashlib.sha256(original).hexdigest(), expected)
            transported = original.replace(b'\r\n', b'\n')
            self.assertNotEqual(hashlib.sha256(transported).hexdigest(), expected)
            manifest.write_bytes(transported)
            metadata = (reference/'bridge_metadata.json').read_bytes()
            data_root = root/'fake-data'
            data_dir = data_root/'cifar-100/cifar-100-python'
            data_dir.mkdir(parents=True)
            for name in ('train', 'test', 'meta'):
                (data_dir/name).write_bytes(b'synthetic file-presence fixture; never used for training')
            args = launcher.parse_args(['--reference-run', str(reference), '--data-root', str(data_root),
                                       '--output-root', str(root/'runs')])
            jobs = launcher.plan_jobs(args)
            launcher.merge_plan(args, jobs)
            def assert_before_gpu(command, **kwargs):
                run = Path(command[command.index('--output-dir')+1])
                payload = (run/'protocol/probe_manifest.csv').read_bytes()
                source = protocol.load_json(run/'protocol/source_metadata.json')
                # This is the exact original BridgeAudit/LAControl comparison that failed.
                self.assertEqual(hashlib.sha256(payload).hexdigest(), source['probe_manifest_sha256'])
                self.assertEqual(payload, original)
                self.assertEqual((run/'protocol/source_metadata.json').read_bytes(), metadata)
                self.assertEqual(protocol.load_json(run/'probe_manifest_replay.json')['newline_conversion'], 'CRLF')
            with patch.object(launcher.subprocess, 'run', side_effect=assert_before_gpu) as execute, \
                    patch('tools.sfra.simple_controls_summary.read_result'):
                for job in jobs:
                    launcher.execute_job(job, args.output_root)
                self.assertEqual(execute.call_count, 4)
            self.assertEqual(manifest.read_bytes(), transported)
            self.assertEqual((reference/'bridge_metadata.json').read_bytes(), metadata)

    def test_reference_content_mismatch_fails_before_registering_training(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp)/'reference'
            shutil.copytree(REPO/'references/full10_clientlt', reference)
            path = reference/'protocol/probe_manifest.csv'
            rows = protocol.read_csv(path)
            rows[0]['raw_train_index'] = int(rows[0]['raw_train_index'])+1
            protocol.write_table(path, rows)
            args = launcher.parse_args(['--reference-run', str(reference)])
            with self.assertRaisesRegex(ValueError, 'probe_manifest_sha256 mismatch'):
                launcher.plan_jobs(args)

    def test_real_partition_and_four_seed42_jobs(self):
        _, counts = protocol.protocol_info(REPO/'references/full10_clientlt')
        self.assertEqual(sum(counts['sizes']), 10847)
        self.assertEqual(sum(counts['tail_counts']), 153)
        self.assertEqual(counts['tail_ids'], list(range(80, 100)))
        self.assertEqual((min(counts['coverage']), max(counts['coverage'])), (2, 27))
        args = launcher.parse_args([])
        jobs = launcher.plan_jobs(args)
        self.assertEqual([j['method'] for j in jobs], list(protocol.NEW_METHODS))
        for job in jobs:
            self.assertEqual(job['seed'], 42)
            cmd = launcher.command_for(job, args.output_root)
            self.assertEqual(cmd[cmd.index('--sfra_variant')+1], protocol.implementation_variant(job['method']))
            self.assertEqual(cmd[cmd.index('--sfra_retention_weight')+1], '10')
            self.assertEqual(cmd[cmd.index('--gamma')+1], '1')  # Existing optimizer gamma stays fixed.
            self.assertNotIn('--sfra_b_transfer', cmd)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            launcher.parse_args(['--seed', '3407'])

    def test_actual_weights_in_b_and_a_aggregation_preserve_original_q(self):
        with tempfile.TemporaryDirectory() as temp:
            rt = object.__new__(Runtime)
            rt.root = Path(temp)
            rt.control_method = 'tailrw-g4'
            rt.sfra_config = {}
            sizes, tails = list(range(1, 31)), [0]*15+[1]*15
            rt.q = dict(enumerate(protocol.tail_weights(sizes, tails, 0)))
            original = dict(rt.q)
            rt.control_weights = protocol.tail_weights(sizes, tails, 4)
            state = dict(proj_lora_A=torch.tensor([3.]), proj_lora_B=torch.tensor([5.]), frozen=torch.tensor([7.]))
            for rnd, factor, extra in ((1, 'B', False), (100, 'B', False), (1, 'A', True), (90, 'A', True)):
                weights = rt.phase_aggregation_weights(list(range(30)), rnd, factor, extra)
                key = 'proj_lora_'+factor
                uploads = {j: {key: torch.tensor([float(j)])} for j in range(30)}
                if extra:
                    result = aggregate_refresh_deltas(state, uploads, weights, [key])
                    expected = float(state[key])+sum(j*w for j, w in enumerate(rt.control_weights))
                else:
                    result = aggregate_lora_state(state, uploads, list(range(30)), [key], weights)
                    expected = sum(j*w for j, w in enumerate(rt.control_weights))
                self.assertAlmostEqual(float(result[key]), expected, places=5)
                for k in state:
                    if k != key:
                        torch.testing.assert_close(result[k], state[k], rtol=0, atol=0)
                self.assertEqual(rt.q, original)
            with self.assertRaises(ValueError):
                rt.phase_aggregation_weights(list(range(30)), 91, 'A', True)
            rt.control_method = 'cover-cp'
            self.assertEqual(rt.phase_aggregation_weights(list(range(30)), 1, 'B'), original)

    def test_cover_changes_only_weights_including_history_only_and_inactive(self):
        rt = object.__new__(Runtime)
        rt.variant, rt.control_method = 'full-cp', 'cover-cp'
        rt.control_counts = dict(coverage=[2, 5, 10])
        rt.bank = SimpleNamespace(tokens=[dict(class_id=c) for c in (0, 0, 1, 2, 2)])
        scores = torch.tensor([[-.1, -.2]]*5)
        source = dict(supported=torch.tensor([True, False, True, False, False]),
                      u=torch.tensor([.3, 0., .2, 0., 0.]), cache=torch.tensor([1., .2, .5, .9, .1]))
        valid = torch.tensor([False, True, False, True, False])
        history = torch.tensor([0., .2, 0., .4, 0.])
        before = copy.deepcopy(source)
        full = SFRARuntime.build_refresh_targets(rt, scores, source, history, valid, 1)
        cover = rt.build_refresh_targets(scores, source, history, valid, 1)
        for key in full:
            if key != 'weights':
                torch.testing.assert_close(full[key], cover[key], rtol=0, atol=0)
        torch.testing.assert_close(cover['weights'], torch.tensor([5/13, 5/13, 2/13, 1/13, 0.]))
        for key in source:
            torch.testing.assert_close(source[key], before[key], rtol=0, atol=0)
        source['supported'].fill_(False)
        empty = rt.build_refresh_targets(scores, source, history, torch.zeros_like(valid), 1)
        self.assertEqual(empty['weights'].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(empty['weights']).all())

    def test_real_cp_refresh_common_state_changes_priority_only_and_checkpoints(self):
        outcomes = []
        for method in ('full-cp', 'cover-cp'):
            bank = visual_bank(True)
            rt = object.__new__(Runtime)
            rt.trainer = SimpleNamespace(model=bank.model, device=bank.device)
            rt.bank, rt.a_keys = bank, bank.keys
            rt.b_keys = sorted(n for n, _ in bank.model.named_parameters() if n.endswith('_lora_B'))
            rt.keys = sorted(rt.a_keys+rt.b_keys)
            rt.variant, rt.control_method = 'full-cp', method
            rt.strength, rt.scaling, rt.classification_strength = 10., .5, 1.
            rt.control_counts = dict(coverage=[2, 5, 10])
            rt.costs, rt.events, rt.q = [], [{}], {0: .8, 1: .2}
            middle = copy.deepcopy(bank.model.state_dict())
            rt.history = FunctionalHistory(bank.evaluate()['scores'], 2)
            rt.history.valid[:] = True
            rt.history.level[:] = 1.3
            torch.manual_seed(318)
            deltas = {j: {k: torch.randn_like(middle[k])*.003 for k in rt.a_keys} for j in (0, 1)}
            ordinary = aggregate_refresh_deltas(middle, deltas, rt.q, rt.a_keys)
            with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
                rt.root = Path(temp)
                committed, _, arrays, summary = rt.refresh(middle, 1, rt.root, prepared=(ordinary, deltas))
                self.assertEqual(summary['correction_steps'], 3)
                self.assertLessEqual(summary['correction_a_norm'], summary['radius']+1e-6)
                for k in middle:
                    if k not in rt.a_keys:
                        torch.testing.assert_close(committed[k], middle[k], rtol=0, atol=0)
                self.assertEqual(len(protocol.read_csv(rt.root/'functional_priority/r001.csv')), 3)
                rt.base_state = middle
                rt.sfra_config = dict(variant='full-cp', simple_control=protocol.control_config(method))
                rt.budget, rt.evaluations, rt.summaries = [], [], [summary]
                rt.b_transfer, rt.elapsed_before, rt.started = None, 0., 0.
                rt.flush_records = lambda completed: None
                (rt.root/'checkpoints').mkdir()
                rt.checkpoint(committed, 1)
                saved = torch.load(rt.root/'checkpoints/sfra_last.pt', weights_only=False)
                self.assertEqual(saved['sfra_config'], rt.sfra_config)
                for key, value in rt.history.state_dict().items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(saved['functional_history'][key], value, rtol=0, atol=0)
                outcomes.append((arrays, summary))
        a, b = outcomes
        for key in ('F_post_B', 'F_proposal', 'responses', 'source_cache', 'current_target', 'target',
                    'active', 'sigma', 'history_before', 'history_valid_before',
                    'classification_proposal_scores', 'classification_global_token_weights'):
            torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)
        for key in ('radius', 'classification_reference_loss', 'classification_scale'):
            self.assertEqual(a[1][key], b[1][key])
        self.assertFalse(torch.equal(a[0]['weights'], b[0]['weights']))

    def test_official_predictions_preserve_model_rng_and_sample_order(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            rt = object.__new__(Runtime)
            rt.root, rt.control_method = Path(temp), 'tailrw-g4'
            rt.args, rt.evaluations = SimpleNamespace(partition='client-longtail'), []
            rt.head, rt.middle, rt.tail = np.arange(20), np.arange(20, 80), np.arange(80, 100)
            labels = torch.arange(100).repeat_interleave(100)
            model = torch.nn.Linear(1, 1)
            model.train()
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(labels, labels), batch_size=500)
            def infer(images):
                torch.rand(3)  # Evaluation must not advance the training RNG.
                return torch.nn.functional.one_hot(images, 100).float()
            rt.global_trainer = SimpleNamespace(model=model, test_loader=loader,
                parse_batch_test=lambda batch: batch, model_inference=infer,
                dm=SimpleNamespace(dataset=SimpleNamespace(data_test=[SimpleNamespace(label=int(c)) for c in labels])))
            before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
            state = {k: v+1 for k, v in before.items()}
            rt.publish(state, 0, 0)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key in before:
                torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)
            with np.load(rt.root/'predictions/r000.npz') as z:
                np.testing.assert_array_equal(z['sample_id'], np.arange(10000))
                np.testing.assert_array_equal(z['class_id'], labels.numpy())
                self.assertTrue(z['correct'].all())
            row = protocol.read_csv(rt.root/'round_metrics.csv')[0]
            self.assertEqual(float(row['bottom20_tail_acc']), 100.)
            self.assertEqual(len(rt.evaluations), 1)

    def test_plan_rejects_changed_gamma_code_and_resume_command(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            args = launcher.parse_args(['--output-root', temp])
            jobs = launcher.plan_jobs(args)
            launcher.merge_plan(args, jobs)
            changed = copy.deepcopy(jobs)
            changed[0]['code_sha256']['utils/sfra_math.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'configuration/source changed'):
                launcher.merge_plan(args, changed)
            job, root = jobs[0], Path(temp)
            run = root/job['run']
            (run/'checkpoints').mkdir(parents=True)
            (run/'checkpoints/sfra_last.pt').write_bytes(b'checkpoint')
            write_json(run/'simple_control_job.json', job)
            command = launcher.command_for(job, root)
            write_json(run/'command.json', command)
            with patch.object(launcher, 'preflight_job'), patch.object(launcher, 'validate_config'), \
                    patch.object(launcher.subprocess, 'run') as launch, \
                    patch('tools.sfra.simple_controls_summary.read_result'):
                with self.assertRaisesRegex(ValueError, 'use --resume'):
                    launcher.execute_job(job, root)
                launcher.execute_job(job, root, resume=True)
                actual = launch.call_args.args[0]
                self.assertEqual(Path(actual[actual.index('--sfra_resume')+1]), (run/'checkpoints/sfra_last.pt').resolve())
                command[command.index('--sfra_retention_weight')+1] = '11'
                write_json(run/'command.json', command)
                with self.assertRaisesRegex(ValueError, 'Saved original command differs'):
                    launcher.execute_job(job, root, resume=True)

    def test_prediction_transitions_and_missing_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = dict(run=root, performance=dict(method='cover-cp'), is_new=True,
                counts=dict(tail_ids=list(range(80, 100))),
                groups=dict(bottom20_tail_acc=list(range(80, 100))),
                curves=[dict(bottom20_tail_acc=50. if rnd == 0 else 50.05) for rnd in range(101)])
            with self.assertRaisesRegex(ValueError, '101'):
                sample_metrics(result)
            (root/'predictions').mkdir()
            labels = np.repeat(np.arange(100), 100)
            initial = labels.copy(); initial[1::2] = (initial[1::2]+1) % 100
            final = initial.copy(); final[8000] = 81; final[[8001, 8003]] = 80
            for rnd in range(101):
                pred = initial if rnd == 0 else final
                np.savez_compressed(root/'predictions'/f'r{rnd:03d}.npz', sample_id=np.arange(10000),
                    class_id=labels, prediction=pred, correct=labels == pred)
            rows, _, signature = sample_metrics(result)
            self.assertTrue(signature)
            first = next(r for r in rows if r['round'] == 1 and r['group'] == 'Tail20')
            self.assertEqual((first['retained'], first['learned'], first['correct_to_wrong'], first['wrong_to_correct']),
                             (999, 2, 1, 2))
            (root/'predictions/r050.npz').unlink()
            result['is_new'] = False
            self.assertEqual(sample_metrics(result), ([], 'unavailable in reused archive', None))

    def test_portable_pack_keeps_predictions_and_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root, reference = Path(temp)/'suite', Path(temp)/'legacy'
            reference.mkdir(); root.mkdir()
            write_json(reference/'completion.json', {'complete': True})
            write_json(root/protocol.PLAN_NAME, dict(contract=protocol.CONTRACT, jobs=[], baselines={'s': str(reference)}))
            (root/'runs/seed42/cover-cp/predictions').mkdir(parents=True)
            (root/'runs/seed42/cover-cp/predictions/r000.npz').write_bytes(b'prediction')
            (root/'runs/seed42/cover-cp/checkpoints').mkdir()
            (root/'runs/seed42/cover-cp/checkpoints/state.json').write_text('{}')
            destination = pack(root, REPO)
            with tarfile.open(destination) as archive:
                names = archive.getnames()
                self.assertIn('suite/runs/seed42/cover-cp/predictions/r000.npz', names)
                self.assertFalse(any('/checkpoints/' in n for n in names))
                self.assertIn('suite/references/s/completion.json', names)
                import json
                plan = json.load(archive.extractfile('suite/'+protocol.PLAN_NAME))
                self.assertEqual(plan['baselines'], {'s': 'references/s'})

    def test_completed_control_audit_rejects_wrong_actual_weights_and_extra_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, counts = protocol.protocol_info(REPO/'references/full10_clientlt')
            job = dict(contract=protocol.CONTRACT, method='tailrw-g4', seed=42, counts=counts,
                       code_sha256=protocol.code_hashes(REPO))
            base = dict(seed=42, protocol_seed=42, topology='client-longtail', la_tau=1.,
                normal_steps_expected=105600, extra_steps_expected=31680,
                head_ids=list(range(20)), middle_ids=list(range(20, 80)))
            write_json(root/'simple_control_job.json', job)
            write_json(root/'simple_control_receipt.json', dict(schema_version=protocol.SCHEMA, code_unchanged=True))
            write_json(root/'sfra_config.json', dict(protocol.FROZEN, variant='s', seed=42, protocol_seed=42,
                partition='client-longtail', witness_batch_size=8, base_training_config=base,
                simple_control=protocol.control_config('tailrw-g4')))
            write_json(root/'execution_config.json', dict(version=2, feedback_forward_batch_size=128, device_cache_gib=4.))
            write_json(root/'completion.json', dict(completed_round=100, official_test_passes=101,
                variant='s', experiment='tailrw-g4', normal_optimizer_steps=105600,
                extra_optimizer_steps=31680, functional_correction_steps=0))
            write_json(root/'class_prior.json', dict(counts=counts['class_counts'], tail_ids=counts['tail_ids']))
            write_json(root/'bridge_metadata.json', {k: 'synthetic-test-'+k for k in protocol.PAIR_HASHES})
            protocol.write_table(root/'partition_manifest.csv',
                protocol.read_csv(REPO/'references/full10_clientlt/partition_manifest.csv'))
            protocol.write_table(root/'round_metrics.csv',
                [dict(round=r, **{m: 70. for m in protocol.METRICS}) for r in range(101)])
            for rnd in range(81, 101):
                protocol.write_table(root/f'per_class_accuracy_epoch_{rnd-1}.csv',
                    [dict(class_id=c, per_class_acc=70.) for c in range(100)])
            rt = object.__new__(Runtime)
            rt.root, rt.control_method = root, 'tailrw-g4'
            rt.control_weights = protocol.tail_weights(counts['sizes'], counts['tail_counts'], 4)
            rt.q = dict(enumerate(protocol.tail_weights(counts['sizes'], counts['tail_counts'], 0)))
            for rnd, factor in [(r, 'B') for r in range(1, 101)]+[(r, 'A') for r in range(1, 91)]:
                rt.phase_aggregation_weights(list(range(30)), rnd, factor, factor == 'A')
            result = read_result(root, 'tailrw-g4', job)
            self.assertEqual(result['performance']['last20_bottom20_tail_acc'], 70.)
            weight_file = root/'aggregation_weights/r090_A.csv'
            rows = protocol.read_csv(weight_file)
            original = rows[0]['weight']
            rows[0]['weight'] = .5
            protocol.write_table(weight_file, rows)
            with self.assertRaisesRegex(ValueError, 'Actual aggregation weight'):
                read_result(root, 'tailrw-g4', job)
            rows[0]['weight'] = original
            protocol.write_table(weight_file, rows)
            protocol.write_table(root/'aggregation_weights/r091_A.csv', rows)
            with self.assertRaisesRegex(ValueError, 'aggregation phases'):
                audit_control_logs(root, 'tailrw-g4', counts)

    def test_available_legacy_results_are_paired_without_fabricated_sample_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launcher.parse_args(['--output-root', temp])
            baselines = launcher.baseline_paths(args)
            if len(baselines) != 2:
                self.skipTest('Optional legacy result archive is absent')
            a, s = [read_result(baselines[m], m) for m in ('full-cp', 's')]
            self.assertEqual(a['signature'], s['signature'])
            write_json(Path(temp)/protocol.PLAN_NAME, dict(contract=protocol.CONTRACT, jobs=[], baselines=baselines))
            status = summarize(temp)
            self.assertFalse(any(r['status'] in ('invalid', 'mismatch') for r in status))
            paired = protocol.read_csv(Path(temp)/'analysis/paired_per_seed.csv')
            self.assertEqual(len(paired), 1)
            self.assertEqual(int(paired[0]['repetitions']), 1)
            self.assertAlmostEqual(float(paired[0]['delta_bottom20_tail_acc']), 2.2725)


if __name__ == '__main__':
    unittest.main()
