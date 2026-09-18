"""CPU-only partition/command checks; no model construction or training."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import cliplora_fresh_protocol
from scripts import run_cliplora_la_control as launcher
from scripts import run_cliplora_standard_dirichlet_reruns as reruns
from scripts.run_cliplora_a_refresh import build_command
from utils.datasplit import partition_fine_class_dirichlet


class StandardDirichletProtocolTests(unittest.TestCase):
    def test_actual_partition_preserves_class_counts_not_clientlt_capacities(self):
        counts = np.array([int(500*.01**(c/99)) for c in range(100)])
        labels = np.repeat(np.arange(100),counts)
        test_labels = np.repeat(np.arange(100),100)
        first,_ = partition_fine_class_dirichlet(labels,test_labels,30,100,.5,42)
        second,_ = partition_fine_class_dirichlet(labels,test_labels,30,100,.5,42)
        self.assertEqual(int(counts.sum()),10847)
        self.assertEqual(sorted(np.concatenate(list(first.values())).tolist()),list(range(10847)))
        matrix = np.array([np.bincount(labels[first[k]],minlength=100) for k in range(30)])
        np.testing.assert_array_equal(matrix.sum(0),counts)
        sizes = matrix.sum(1).tolist()
        self.assertNotEqual(sizes[-3:],[46,39,45])
        self.assertGreaterEqual(min(sizes),10)
        for k in range(30):
            np.testing.assert_array_equal(first[k],second[k])

    def test_configuration_names_keep_historical_paths_and_separate_fine_beta(self):
        args = SimpleNamespace(method='e3',la_tau=1.,a_lr_mult=1.,protocol_seed=42,
                               partition='matched-dirichlet',dirichlet_beta=.5)
        self.assertEqual(launcher.configuration_id(args),'tau1_a1_protocol42')
        args.partition='noniid-labeldir-fine'
        self.assertEqual(launcher.configuration_id(args),'tau1_a1_protocol42_beta0.5')

    def test_build_command_uses_fine_and_separate_output(self):
        args = SimpleNamespace(output_root=Path('output/pilot'),seed=42,data_root=Path('DATA'),
                               schedule_file=None,partition='noniid-labeldir-fine',matched_beta=.5,
                               rank=4,refresh_interval=10,refresh_epochs=1,refresh_lr=.001,
                               resume=None,num_workers=8)
        command, output = build_command(args,'c2')
        self.assertEqual(command[command.index('--partition')+1],'noniid-labeldir-fine')
        self.assertEqual(command[command.index('--beta')+1],'0.5')
        self.assertIn('noniid-labeldir-fine',output.parts)

    def test_fresh_protocol_reuses_schedule_but_never_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root/'bridge/seed42/client-longtail/c1'
            (reference/'protocol').mkdir(parents=True)
            schedule = [list(range(30)) for _ in range(100)]
            (reference/'protocol/full_schedule.json').write_text(json.dumps({'schedule':schedule}))
            (reference/'partition_manifest.csv').write_text('do not copy client assignments')
            (reference/'bridge_metadata.json').write_text(json.dumps({'client_sample_counts':[46,39,45]}))
            args = SimpleNamespace(seed=42,protocol_seed=42,partition='noniid-labeldir-fine',
                                   dirichlet_beta=.5,data_root=root/'DATA',bridge_root=root/'bridge',
                                   schedule_file=None)
            def build_protocol(destination,**kwargs):
                Path(destination).mkdir(parents=True)
                (Path(destination)/'eri_protocol.json').write_text('{}')
            fake_protocol = SimpleNamespace(build_protocol=build_protocol)
            with patch.dict(sys.modules,{'tools.eri_closure.protocol':fake_protocol}):
                result = cliplora_fresh_protocol.prepare_fresh_protocol(args,root/'run')
            self.assertEqual(json.loads(result.read_text())['schedule'],schedule)
            self.assertFalse((result.parent/'partition_source.csv').exists())
            self.assertFalse((result.parent/'source_metadata.json').exists())
            self.assertTrue((result.parent/'reference_metadata.json').exists())
            protocol=json.loads((result.parent/'partition_protocol.json').read_text())
            self.assertFalse(protocol['reuse_clientlt_capacities'])

    def test_fine_la_launcher_needs_no_bridge_and_passes_no_replay_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def prepare(args,run):
                (run/'protocol').mkdir(parents=True)
                return run/'protocol/full_schedule.json'
            argv=['run','--method','e3','--partition','noniid-labeldir-fine',
                  '--output-root',str(root/'runs'),'--bridge-root',str(root/'missing_bridge')]
            with patch.object(sys,'argv',argv), \
                 patch.object(cliplora_fresh_protocol,'prepare_fresh_protocol',side_effect=prepare), \
                 patch.object(launcher.subprocess,'run') as child, \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            command=child.call_args.args[0]
            self.assertEqual(command[command.index('--lac_partition_manifest')+1],'')
            self.assertEqual(command[command.index('--partition')+1],'noniid-labeldir-fine')
            self.assertEqual(command[command.index('--split_seed')+1],'42')

    def test_sequential_wrapper_does_not_launch_background_or_matched_runs(self):
        with patch.object(sys,'argv',['run']),patch.object(reruns.subprocess,'run') as child, \
             contextlib.redirect_stdout(io.StringIO()):
            reruns.main()
        self.assertEqual(child.call_count,7)
        self.assertEqual([c.args[0][c.args[0].index('--method')+1] for c in child.call_args_list],
                         ['e3','j','s','e2','e5','c1','c2'])
        for call in child.call_args_list:
            command=call.args[0]
            self.assertEqual(command[command.index('--partition')+1],'noniid-labeldir-fine')
            self.assertEqual(command[command.index('--stage')+1],'train')
            self.assertTrue(call.kwargs['check'])


if __name__=='__main__':
    unittest.main()
