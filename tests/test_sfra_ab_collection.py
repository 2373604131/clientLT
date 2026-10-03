"""Real round-CSV metadata must survive offline collection of frozen AB runs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tarfile

from scripts import collect_ab_validation as collector
from scripts import run_ab_validation as suite
from tools.sfra import ab_validation as frozen
from tools.sfra.calibration_reference import file_hash
from tools.sfra.summary import read_csv, write_csv
from test_sfra_ab_validation import args, completed


class TestABCollection(unittest.TestCase):
    def test_real_csv_columns_pack_completed_seeds_without_changing_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            arms=('s','a','ab','a-calibration','a-flat','a-current')
            jobs=suite.plan_jobs(args(root,arms,(42,0,3407)))
            suite.merge_plan(root,jobs)
            offsets={'s':0.,'a':.3,'ab':.5,'a-calibration':.4,'a-flat':.2,'a-current':.1}
            saved=[root/frozen.PLAN_NAME]
            for job in jobs:
                if job['seed']==0:
                    if job['arm']=='a':
                        run=root/job['run'];run.mkdir(parents=True)
                        suite.write_json(run/'progress.json',dict(completed_round=23))
                        saved.append(run/'progress.json')
                    continue
                run=completed(root,job,offsets[job['arm']])
                curve=read_csv(run/'round_metrics.csv')
                for row in curve:
                    # These metadata fields are written by LAControlRuntime.official().
                    row.update(epoch=int(row['round'])-1,decision_round=row['round'],
                        method='sfra_'+frozen.ARMS[job['arm']],partition=job['partition'],seed=str(job['seed']))
                write_csv(run/'round_metrics.csv',curve)
                saved.extend(run/name for name in ['round_metrics.csv','sfra_config.json','ab_validation_run.json','completion.json'])
            before={str(p):file_hash(p) for p in saved}
            sources_before=frozen.code_hashes(suite.REPO)

            # Reproduce the reported v1 failure, then use the separate offline entry.
            with self.assertRaisesRegex(TypeError,'multiple values'):
                frozen.summarize(root)
            archive=root.parent/(root.name+'_analysis.tar.gz')
            try:
                with patch('sys.argv',['collect_ab_validation.py','--stage','pack','--output-root',str(root)]),patch('builtins.print'):
                    collector.main()
                curves=read_csv(root/'analysis/curves.csv')
                self.assertEqual(len(curves),12*101)
                self.assertEqual({x['seed'] for x in curves},{'42','3407'})
                for row in curves:
                    self.assertEqual(row['partition'],'client-longtail')
                    self.assertEqual(row['method'],'sfra_'+frozen.ARMS[row['arm']])
                    self.assertEqual(int(row['epoch']),int(row['round'])-1)
                    expected=50+int(row['round'])/100+(offsets[row['arm']] if int(row['round'])>=30 else 0)
                    self.assertAlmostEqual(float(row['bottom20_tail_acc']),expected)
                status=read_csv(root/'analysis/status.csv')
                self.assertEqual(sum(x['status']=='complete' for x in status),12)
                self.assertEqual(next(x['status'] for x in status if x['seed']=='0' and x['arm']=='a'),'incomplete')
                pairs=read_csv(root/'analysis/paired_per_seed.csv')
                ab_a=[x for x in pairs if x['comparison']=='ab minus a']
                self.assertEqual({x['seed'] for x in ab_a},{'42','3407'})
                self.assertTrue(all(abs(float(x['delta_last20_bottom20_tail_acc'])-.2)<1e-9 for x in ab_a))
                summaries=read_csv(root/'analysis/paired_summary.csv')
                confirm=next(x for x in summaries if x['comparison']=='ab minus a' and x['cohort']=='confirmation')
                self.assertEqual(confirm['seeds'],'3407')
                self.assertEqual(confirm['n'],'1')
                self.assertEqual(confirm['delta_last20_bottom20_tail_acc_sd'],'')
                with tarfile.open(archive) as f:
                    self.assertIn(root.name+'/analysis/report.md',f.getnames())
                    self.assertIn(root.name+'/analysis/curves.csv',f.getnames())
                    self.assertFalse(any(n.endswith('.pt') for n in f.getnames()))
                self.assertEqual({str(p):file_hash(p) for p in saved},before)
                self.assertEqual(frozen.code_hashes(suite.REPO),sources_before)
            finally:
                archive.unlink(missing_ok=True)

    def test_invalid_run_is_still_excluded_and_reported_as_failure_after_packing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=suite.plan_jobs(args(root,('a','ab'),(42,)))
            suite.merge_plan(root,jobs)
            for job in jobs:
                run=completed(root,job)
                if job['arm']=='ab':
                    config=frozen.load_json(run/'execution_config.json')
                    config['feedback_forward_batch_size']=64
                    suite.write_json(run/'execution_config.json',config)
            archive=root.parent/(root.name+'_analysis.tar.gz')
            try:
                with patch('sys.argv',['collect_ab_validation.py','--stage','pack','--output-root',str(root)]),patch('builtins.print'),patch('sys.stderr'):
                    with self.assertRaises(SystemExit) as error:
                        collector.main()
                self.assertEqual(error.exception.code,2)
                self.assertTrue(archive.exists())
                status=read_csv(root/'analysis/status.csv')
                self.assertEqual(next(x['status'] for x in status if x['arm']=='ab'),'invalid')
                self.assertEqual(read_csv(root/'analysis/paired_per_seed.csv'),[])
            finally:
                archive.unlink(missing_ok=True)


if __name__=='__main__':
    unittest.main()
