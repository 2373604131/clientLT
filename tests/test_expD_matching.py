import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments.expD_utils import build_matching_candidates, load_approved_pairs, write_json
from experiments.run_expD_matched_response import build_command, schedule_file


def _summary_row(protocol, value, top1, neff, entropy=0.5, cv=0.1):
    key = f"lambda={value:.2f}" if protocol == "clientlt" else f"beta={value:.2f}"
    return {
        "protocol": protocol,
        "parameter_key": key,
        "parameter_value": value,
        "tail_top1_mass_mean_mean": top1,
        "tail_effective_client_number_mean_mean": neff,
        "tail_normalized_entropy_mean_mean": entropy,
        "client_sample_count_cv_mean": cv,
    }


class ExpDMatchingTest(unittest.TestCase):
    def test_matching_distance_ranks_nearest_candidate(self):
        rows = [
            _summary_row("clientlt", 0.5, 0.50, 3.0),
            _summary_row("dirichlet", 0.1, 0.49, 3.1),
            _summary_row("dirichlet", 1.0, 0.20, 8.0),
        ]
        candidates, one_to_one = build_matching_candidates(rows, target_lambdas=[0.5], top_k=2, alpha_t=0.1)

        self.assertEqual(candidates[0]["pair_candidate_rank"], 1)
        self.assertEqual(candidates[0]["dirichlet_beta"], 0.1)
        self.assertLess(candidates[0]["match_distance"], candidates[1]["match_distance"])
        self.assertTrue(candidates[0]["match_ok"])
        self.assertEqual(one_to_one[0]["clientlt_lambda"], 0.5)
        self.assertEqual(one_to_one[0]["dirichlet_beta"], 0.1)

    def test_approved_pairs_disabled_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved_pairs_template.json"
            write_json(
                path,
                {
                    "experiment": "ExpD matched Dirichlet vs Client-LT",
                    "status": "REQUIRES_USER_APPROVAL",
                    "pairs": [
                        {
                            "pair_id": "mild",
                            "enabled": False,
                            "clientlt": {"lambda_T": 0.25, "alpha_T": 0.1},
                            "dirichlet": {"beta": 0.5},
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "No enabled=true pairs"):
                load_approved_pairs(path)

    def test_enabled_pair_dry_run_command_uses_shared_seeds_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                datadir="./DATA",
                output_dir=str(Path(tmp) / "matched_response"),
                dataset="cifar100_LT",
                dataset_config_file="",
                methods=["promptfl", "capt"],
                seeds=[1],
                split_seed=None,
                train_seed=None,
                schedule_seed=None,
                num_clients=50,
                frac=0.2,
                rounds=100,
                global_eval_interval=5,
                tail_client_ratio=0.1,
                tail_class_ratio=0.2,
                imb_factor=0.01,
                imb_type="exp",
                num_classes=100,
                lr=0.001,
                gamma=1.0,
                beta=1.0,
                n_ctx=4,
                n_general=1,
                train_batch_size=32,
                test_batch_size=64,
                num_workers=4,
                ctx_init="False",
                csc="True",
                promptfl_config="configs/trainers/PromptFL/vit_b16.yaml",
                capt_config="configs/trainers/CAPT/vit_b16.yaml",
                clip_config="configs/trainers/PromptFL/vit_b16.yaml",
                federated_entry="federated_main.py",
                python="python",
                gpu="",
                extra_opts=[],
            )
            pair = {
                "pair_id": "mild",
                "enabled": True,
                "clientlt": {"lambda_T": 0.25, "alpha_T": 0.1},
                "dirichlet": {"beta": 0.5},
            }

            prompt_cmd = build_command(args, pair, "clientlt", "promptfl", 1)
            capt_cmd = build_command(args, pair, "clientlt", "capt", 1)
            sched = str(schedule_file(args, "mild", 1))

            self.assertIn("--split_seed", prompt_cmd)
            self.assertEqual(prompt_cmd[prompt_cmd.index("--split_seed") + 1], "1")
            self.assertEqual(capt_cmd[capt_cmd.index("--split_seed") + 1], "1")
            self.assertEqual(prompt_cmd[prompt_cmd.index("--client_schedule_file") + 1], sched)
            self.assertEqual(capt_cmd[capt_cmd.index("--client_schedule_file") + 1], sched)
            self.assertEqual(prompt_cmd[prompt_cmd.index("--client_schedule_seed") + 1], "1")
            self.assertEqual(capt_cmd[capt_cmd.index("--client_schedule_seed") + 1], "1")


if __name__ == "__main__":
    unittest.main()
