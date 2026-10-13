"""Real-pool protocol checks and launch integration; no downloaded weights/GPU."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_dirichlet_benchmarks as launcher
from scripts.run_paper_benchmarks import command_for
from tools.benchmarks import common, collect, dirichlet

SOURCE = common.REPO / 'references/full10_clientlt'


class DirichletBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def reference(self):
        return dirichlet.prepare_reference(SOURCE, self.root / 'reference', .5)

    def job(self, method, reference, smoke=False):
        with patch.object(common, 'source_hashes', return_value={'fixture': 'same'}):
            return common.make_job(method, reference, self.root / 'data', self.root / 'runs',
                                   workers=0, smoke=smoke, suite='dirichlet')

    def test_preserves_actual_images_global_counts_and_source_but_changes_clients(self):
        before = common.reference_contract(SOURCE)
        ref = self.reference()
        old = common.manifest_rows(SOURCE / 'partition_manifest.csv')
        new = common.manifest_rows(ref / 'partition_manifest.csv')
        sample = lambda rows: sorted((r['raw_sample_id'], r['class_id']) for r in rows)
        self.assertEqual(sample(old), sample(new))
        self.assertEqual(len(new), 10847)
        self.assertNotEqual([r['client_id'] for r in sorted(old, key=lambda r:r['raw_sample_id'])],
                            [r['client_id'] for r in sorted(new, key=lambda r:r['raw_sample_id'])])
        sizes, counts = dirichlet.partition_statistics(new)
        self.assertEqual((min(sizes), max(sizes)), (229, 500))
        self.assertEqual(counts, dirichlet.partition_statistics(old)[1])
        self.assertEqual(before, common.reference_contract(SOURCE))
        self.assertEqual(common.read_json(ref / 'bridge_metadata.json')['initial_lora_sha256'],
                         common.read_json(SOURCE / 'bridge_metadata.json')['initial_lora_sha256'])

    def test_repeated_prepare_is_idempotent_and_other_beta_does_not_overwrite(self):
        ref = self.reference()
        before = common.reference_contract(ref)
        self.assertEqual(dirichlet.prepare_reference(SOURCE, ref, .5), ref)
        self.assertEqual(common.reference_contract(ref), before)
        with self.assertRaisesRegex(ValueError, 'Existing Dirichlet'):
            dirichlet.prepare_reference(SOURCE, ref, .1)
        self.assertEqual(common.reference_contract(ref), before)

    def test_rejects_changed_recipe_and_metadata(self):
        ref = self.reference()
        path = ref / 'protocol/partition_protocol.json'
        recipe = common.read_json(path)
        recipe['dirichlet_beta'] = .1
        common.write_json(path, recipe)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            common.reference_contract(ref)

    def test_different_beta_really_generates_different_allocation(self):
        original = common.manifest_rows(SOURCE / 'partition_manifest.csv')
        self.assertNotEqual(dirichlet.partition_rows(original, .5), dirichlet.partition_rows(original, .1))

    def test_groups_and_shards_cover_exactly_fourteen_methods(self):
        self.assertEqual([len(v) for v in dirichlet.GROUPS.values()], [8, 4, 2])
        shards = [dirichlet.select_methods(shard=i, shards=4) for i in range(4)]
        combined = sum(shards, [])
        self.assertEqual(set(combined), set(dirichlet.ALL))
        self.assertEqual(len(combined), 14)
        with self.assertRaises(ValueError):
            dirichlet.select_methods(methods=['fedlf', 'fedlf'])
        with self.assertRaises(ValueError):
            dirichlet.select_methods(shard=4, shards=4)

    def test_all_groups_register_together_with_one_config_and_partition(self):
        ref = self.reference()
        jobs = [self.job(method, ref) for method in dirichlet.ALL]
        for job in jobs:
            common.register(job)
        self.assertEqual(len(list((self.root / 'runs/jobs').glob('*.json'))), 14)
        self.assertEqual(len({common.digest(j['config']) for j in jobs}), 1)
        self.assertEqual(len({common.digest(j['protocol']) for j in jobs}), 1)
        self.assertEqual(jobs[0]['protocol']['topology'], dirichlet.PARTITION)

    def test_ab_commands_and_output_paths_use_dirichlet_and_resume(self):
        ref = self.reference()
        for method in ('a', 'ab'):
            job = self.job(method, ref)
            command = command_for(job, resume=True)
            self.assertEqual(command[command.index('--partition') + 1], dirichlet.PARTITION)
            self.assertEqual(command[command.index('--dirichlet-beta') + 1], '0.5')
            self.assertIn('--resume', command)
            self.assertIn(dirichlet.PARTITION, common.run_path(job).parts)
            self.assertIn('beta0.5', common.run_path(job).name)
        external = command_for(self.job('fedlf', ref), resume=True, stop_after=1)
        self.assertIn('--resume', external)
        self.assertEqual(external[-3:-1], ['--stop-after', '1'])

    def test_runtime_config_uses_new_partition_instead_of_stale_source_path(self):
        from tools.benchmarks.runtime import build_config
        ref = self.reference()
        job = self.job('fedlf', ref)
        cfg = build_config(job, common.read_json(ref / 'bridge_metadata.json'))
        self.assertEqual(cfg.DATASET.PARTITION, dirichlet.PARTITION)
        self.assertEqual(cfg.DATASET.BETA, .5)
        self.assertEqual(cfg.DATASET.SPLIT_SEED, 42)
        self.assertEqual(Path(cfg.DATASET.PARTITION_MANIFEST), ref / 'partition_manifest.csv')

    def test_clientlt_ab_cannot_be_imported_as_dirichlet(self):
        ref = self.reference()
        job = self.job('a', ref)
        run = self.root / 'old_a'
        common.write_json(run / 'ab_validation_run.json', {'job': {
            'seed':42, 'arm':'a', 'protocol_seed':42, 'partition':'client-longtail'}})
        with self.assertRaisesRegex(ValueError, 'topology differs'):
            collect.validate_ab(run, job)

    def test_sfra_replays_exact_new_manifest_and_schedule(self):
        from scripts.run_cliplora_sfra import prepare_protocol
        from types import SimpleNamespace
        ref = self.reference()
        run = self.root / 'sfra'
        schedule, manifest = prepare_protocol(SimpleNamespace(reference_run=ref, partition=dirichlet.PARTITION), run)
        self.assertEqual(Path(manifest).read_bytes(), (ref / 'partition_manifest.csv').read_bytes())
        self.assertEqual(schedule.read_bytes(), (ref / 'protocol/full_schedule.json').read_bytes())
        self.assertEqual(common.read_json(run / 'protocol/source_metadata.json')['topology'], dirichlet.PARTITION)

    def test_plan_never_trains_and_registers_all_methods(self):
        out = self.root / 'plan'
        with patch.object(common, 'source_hashes', return_value={'fixture':'same'}), \
             patch.object(launcher, 'execute') as execute, patch.object(launcher, 'preflight') as preflight, \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main(['--output-root', str(out)])
        execute.assert_not_called()
        preflight.assert_not_called()
        self.assertEqual(len(list((out / 'jobs').glob('*.json'))), 14)

    def test_smoke_is_separate_and_excludes_ours(self):
        out = self.root / 'smoke_suite'
        with patch.object(common, 'source_hashes', return_value={'fixture':'same'}), \
             patch.object(launcher, 'execute') as execute, patch.object(launcher, 'preflight', return_value='ready'), \
             contextlib.redirect_stdout(io.StringIO()):
            launcher.main(['--stage', 'smoke', '--output-root', str(out)])
        self.assertEqual(execute.call_count, 12)
        self.assertTrue(all(c.args[0]['smoke'] for c in execute.call_args_list))
        self.assertTrue(all(Path(c.args[0]['output_root']) == (out / 'smoke').resolve() for c in execute.call_args_list))
        self.assertEqual((out / 'smoke/analysis/performance.csv').read_text().strip(), '')
        self.assertFalse((out / 'jobs').exists())

    def test_partial_suite_summary_does_not_fabricate_scores(self):
        ref = self.reference()
        for method in dirichlet.ALL:
            common.register(self.job(method, ref))
        self.assertEqual(collect.collect(self.root / 'runs'), [])
        statuses = common.read_csv(self.root / 'runs/analysis/status.csv')
        self.assertEqual(len(statuses), 14)
        self.assertTrue(all(row['status'] == 'not_started' for row in statuses))
        self.assertIn('ordinary fine-class Dirichlet',
                      (self.root / 'runs/analysis/report.md').read_text())


if __name__ == '__main__':
    unittest.main()
