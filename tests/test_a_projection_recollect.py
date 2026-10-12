"""Recollection contract/dispatch tests; no synthetic values are paper results."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import run_a_projection_recollect as recollect
from scripts import run_cliplora_joint_aggregation as single
from tools.client_aggregation.protocol import check_fixed_class_groups, load_json
from tools.client_aggregation.runtime import runtime_class

REPO = Path(__file__).resolve().parents[1]


class RecollectTests(unittest.TestCase):
    def test_original_partition_and_fixed_tail20_are_reused(self):
        contract = recollect.reference_contract(REPO / 'references/full10_clientlt')
        self.assertEqual(contract['fixed_class_groups']['tail_ids'], list(range(80, 100)))
        self.assertEqual(contract['training_samples'], 10847)
        self.assertEqual(contract['partition_sha256'], recollect.PARTITION_SHA256)
        self.assertEqual(contract['class_counts'][-20:], [12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5])

    def test_changed_tail_classes_or_sample_assignment_fail_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'reference'
            shutil.copytree(REPO / 'references/full10_clientlt', reference)
            protocol_path = reference / 'protocol/eri_protocol.json'
            original = json.loads(protocol_path.read_text())
            bad = dict(original, tail_class_ids=list(range(70, 90)))
            protocol_path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '80..99'):
                recollect.reference_contract(reference)
            protocol_path.write_text(json.dumps(original), encoding='utf-8')
            partition = reference / 'partition_manifest.csv'
            lines = partition.read_text().splitlines()
            # Swap two raw IDs without altering the class/client count matrix.
            header, first, second = lines[0].split(','), lines[1].split(','), lines[2].split(',')
            col = header.index('raw_sample_id')
            first[col], second[col] = second[col], first[col]
            lines[1], lines[2] = ','.join(first), ','.join(second)
            partition.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'assignment changed'):
                recollect.reference_contract(reference)

    def test_register_one_fedavg_plan_serial_defaults_and_frozen_schedule(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = recollect.parse_args(['--output-root', tmp])
            plan = recollect.register(args)
            self.assertEqual(plan, recollect.register(args))
            self.assertEqual(plan['aggregation']['rule'], 'fedavg')
            self.assertEqual(plan['settings']['client_concurrency'], 1)
            self.assertEqual(plan['settings']['fixed_class_groups']['tail_ids'], list(range(80, 100)))
            self.assertFalse((Path(tmp) / 'tailrw16').exists())
            self.assertFalse((Path(tmp) / 'joint').exists())
            job = single.make_job(plan, 'plain_a')
            command = single.command_for(job, Path(tmp) / 'fedavg')
            self.assertEqual(command[command.index('--tail_class_ratio') + 1], '0.2')
            self.assertEqual(command[command.index('--head_class_ratio') + 1], '0.8')
            self.assertEqual(command[command.index('--round') + 1], '100')
            self.assertEqual(plan['contract']['A_rounds'], list(range(1, 91)))
            self.assertNotIn('--b_transfer_enable', command)
            args.num_workers = 2
            with self.assertRaisesRegex(ValueError, 'Registered code/reference/settings changed'):
                recollect.register(args)

    def test_runtime_rejects_moved_tail_before_any_local_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            plan = recollect.register(recollect.parse_args(['--output-root', tmp]))
            job = single.make_job(plan, 'plain_a')
            rt = object.__new__(runtime_class(job))
            rt.config = dict(recollect.GROUPS, tail_ids=list(range(70, 90)))
            # No trainer/model attached: the grouping guard must reject first.
            with self.assertRaisesRegex(ValueError, 'tail_ids'):
                rt.configure_experiment()
            check_fixed_class_groups(job, recollect.GROUPS)
            check_fixed_class_groups(dict(settings={}), rt.config)  # opt-in only

    def test_one_allocated_gpu_smoke_formal_audit_then_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            calls = []
            def execute(job, root):
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-allocated')
                self.assertEqual(job['settings']['client_concurrency'], 1)
                calls.append(job['mode'])
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-allocated'}), patch.object(
                    single, 'preflight'), patch.object(single, 'execute', new=execute), patch.object(
                    recollect, 'audit_inputs', side_effect=lambda a: calls.append('audit')), patch.object(
                    recollect, 'diagnose', side_effect=lambda a: calls.append('diagnose')):
                recollect.main(['--stage', 'server', '--output-root', tmp])
            self.assertEqual(calls, ['smoke', 'formal', 'audit', 'diagnose'])

    def test_failed_training_never_starts_diagnosis_and_train_only_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), patch.object(single, 'preflight'), patch.object(
                    single, 'execute', side_effect=ValueError('failed smoke')), patch.object(recollect, 'diagnose') as diagnose:
                with self.assertRaisesRegex(ValueError, 'failed smoke'):
                    recollect.main(['--stage', 'server', '--output-root', tmp])
                diagnose.assert_not_called()
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3'}), patch.object(single, 'preflight'), patch.object(
                    single, 'execute'), patch.object(recollect, 'audit_inputs') as audit, patch.object(recollect, 'diagnose') as diagnose:
                recollect.main(['--stage', 'server', '--train-only', '--output-root', tmp])
                audit.assert_called_once()
                diagnose.assert_not_called()

    def test_capture_requires_original_proposals_and_all_event_files(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args = recollect.parse_args(['--output-root', str(root)])
            run = root / 'fedavg/runs/seed42/plain_a'
            for rnd in range(1, 101):
                for phase in (('normal_B', 'refresh_A') if rnd <= 90 else ('normal_B',)):
                    file = run / 'events' / ('r%03d_c000_main_%s/state.pt' % (rnd, phase))
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(b'file presence fixture')
            (run / 'checkpoints').mkdir()
            (run / 'checkpoints/base_model.pt').write_bytes(b'base presence fixture')
            (run / 'completion.json').write_text(json.dumps(dict(completed_round=100)))
            class Store:
                a_keys = ['a']
                def __init__(self, source):
                    pass
                def committed(self, rnd):
                    return {'a': torch.zeros(2)}
                def anchor(self, rnd):
                    return self.committed(rnd), {}
                def historical_proposals(self, rnd, anchor):
                    return {j: {} for j in range(30)}, {'proposals': 'original_historical_uploads'}
            with patch('tools.a_projection_diagnosis.protocol.check_source', return_value=dict(path=str(run))), patch(
                    'tools.a_projection_diagnosis.states.StateStore', Store):
                receipt = recollect.audit_inputs(args)
                self.assertEqual(receipt['event_tensor_count'], 190)
                self.assertEqual([p['clients'] for p in receipt['original_proposals']], [30] * 3)
                with patch.object(Store, 'historical_proposals', return_value=(None, {})):
                    with self.assertRaisesRegex(ValueError, 'all 30 historical proposals'):
                        recollect.audit_inputs(args)
                (run / 'events/r001_c000_main_normal_B/state.pt').unlink()
                with self.assertRaisesRegex(FileNotFoundError, '190'):
                    recollect.audit_inputs(args)


if __name__ == '__main__':
    unittest.main()
