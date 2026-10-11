"""Partition intervention/manifest replay and legacy-trainer integration on CPU."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_cliplora_joint_tail30 import parse_args
from scripts import run_cliplora_joint_aggregation as single
from tools.client_aggregation import tail30
from tools.client_aggregation.protocol import load_json, read_csv, validate_arm


@unittest.skipUnless((tail30.ROOT/'DATA/cifar-100/cifar-100-python/train').exists(), 'CIFAR train file required')
class TestTail30(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.args = parse_args(['--output-root',str(cls.root),'--data-root',str(tail30.ROOT/'DATA')])
        cls.plans = tail30.register(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pool_partition_and_client_positions(self):
        old = read_csv(tail30.ROOT/'references/full10_clientlt/partition_manifest.csv')
        new = read_csv(self.root/'reference/partition_manifest.csv')
        self.assertEqual(sorted((r['raw_sample_id'],r['class_id']) for r in old),
                         sorted((r['raw_sample_id'],r['class_id']) for r in new))
        self.assertNotEqual(old,new)
        for client in range(30):
            positions = [int(r['local_position']) for r in new if int(r['client_id'])==client]
            self.assertEqual(positions,list(range(len(positions))))
        tail_clients = [r for r in new if int(r['client_id'])>=27]
        self.assertEqual(sum(int(r['class_id'])>=70 for r in tail_clients),236)
        self.assertEqual(len(tail_clients),259)
        info=load_json(self.root/'reference/tail30_partition.json')
        self.assertEqual(info['training_tail_samples'],305)
        self.assertEqual(info['steps_per_epoch'],352)

    def test_three_arms_identical_joint_no_extra_methods(self):
        self.assertEqual(set(self.plans),{'frozen','plain_a','method_a'})
        self.assertEqual(len({p['aggregation']['weights_sha256'] for p in self.plans.values()}),1)
        for arm,plan in self.plans.items():
            job=single.make_job(plan,arm)
            command=single.command_for(job,self.root/arm)
            for flag,value in [('--head_class_ratio','0.7'),('--tail_class_ratio','0.3'),
                               ('--local_epochs','3'),('--seed','42'),('--round','100')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertNotIn('--b_transfer_enable',command)
            self.assertEqual(command[command.index('--sfra_variant')+1],'full-cp' if arm=='method_a' else 's')
            self.assertEqual(plan['contract']['normal_B_steps'],105600)
            self.assertEqual(plan['contract']['extra_A_steps'],0 if arm=='frozen' else 31680)
            if arm=='plain_a':
                self.assertEqual(command[command.index('--sfra_retention_weight')+1],'0')
            with self.assertRaises(ValueError):
                single.make_job(plan,'ab')
            with self.assertRaises(ValueError):
                single.make_job(plan,'method_a_v2')
        broken=copy.deepcopy(single.make_job(self.plans['plain_a'],'plain_a'))
        broken['aggregation']['partition_intervention']['partition_parameters']['tail_class_ratio']=.2
        with self.assertRaises(ValueError):
            validate_arm(broken)

    def test_resume_reuses_weights_and_prepared_protocol(self):
        with patch('tools.client_aggregation.tail30.aggregation_spec',side_effect=AssertionError('must not resolve')):
            self.assertEqual(tail30.register(self.args),self.plans)
        from scripts.run_method_a_simple_controls import prepare_control_protocol
        run=self.root/'prepared_test'
        prepare_control_protocol(dict(reference_run=str(self.root/'reference')),run)
        self.assertEqual(read_csv(run/'protocol/partition_source.csv'),
                         read_csv(self.root/'reference/partition_manifest.csv'))
        self.assertEqual(load_json(run/'protocol/source_metadata.json')['client_sample_counts'],
                         self.plans['frozen']['aggregation']['counts']['sizes'])
        self.assertTrue((run/'protocol/tail30_partition.json').exists() is False)

    def test_status_and_missing_controls_do_not_start_training(self):
        self.assertTrue(all(r['status']=='missing' for r in tail30.status(self.root)))
        self.assertTrue(all(r['status']=='missing' for r in tail30.summarize(self.root)))

    def test_budget_contract_uses_new_client_capacities(self):
        counts=copy.deepcopy(self.plans['plain_a']['aggregation']['counts'])
        counts['sizes'][0]+=32
        contract=tail30.contract_for('plain_a',counts)
        self.assertEqual(contract['normal_B_steps'],105900)
        self.assertEqual(contract['extra_A_steps'],31770)


if __name__=='__main__':
    unittest.main()
