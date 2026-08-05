import unittest

import torch

from utils.aggregation_crush import measure_aggregation_crush, should_log_aggregation_crush


class FakeModel:
    """A model whose per-class accuracy is dictated by the loaded state.

    ``load_state_dict`` records a tag, and ``global_test`` (queried via the
    trainer) returns the accuracy table registered for that tag. This lets the
    test script exercise the crush measurement with fully controlled numbers.
    """

    def __init__(self, tables):
        self._tables = tables
        self._current = None
        self.training = True

    def load_state_dict(self, state, strict=False):
        self._current = state["__tag__"]

    def train(self, mode=True):
        self.training = mode


class FakeTrainer:
    def __init__(self, tables):
        self.model = FakeModel(tables)
        self._tables = tables

    def global_test(self, is_global=False, current_epoch=0):
        table = self._tables[self.model._current]
        # Mimic global_test's return: [metrics..., class_accuracy_dict].
        return [0.0, 0.0, 0.0, dict(table)]


def _state(tag):
    return {"__tag__": tag}


class AggregationCrushTest(unittest.TestCase):
    def test_crush_measured_where_mass_share_is_small(self):
        num_users = 4
        num_classes = 5  # class 4 is the single tail class (bottom 20%).
        # Global counts: classes 0-3 large, class 4 tiny -> class 4 is tail.
        client_class_counts = {
            0: torch.tensor([50.0, 0, 0, 0, 0]),
            1: torch.tensor([0, 50.0, 0, 0, 0]),
            2: torch.tensor([0, 0, 50.0, 0, 0]),
            3: torch.tensor([0, 0, 0, 50.0, 4.0]),  # only client 3 holds tail class 4.
        }
        datanumber_client = {0: 50, 1: 50, 2: 50, 3: 54}
        idxs_users = [0, 1, 2, 3]

        # Client 3's LOCAL model nails the tail class; the FedAvg GLOBAL model loses it.
        tables = {
            "local3": {0: 0.0, 1: 0.0, 2: 0.0, 3: 90.0, 4: 80.0},
            "local0": {0: 90.0, 1: 0, 2: 0, 3: 0, 4: 0.0},
            "local1": {0: 0, 1: 90.0, 2: 0, 3: 0, 4: 0.0},
            "local2": {0: 0, 1: 0, 2: 90.0, 3: 0, 4: 0.0},
            "pre": {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 5.0},
            "post": {0: 70.0, 1: 70.0, 2: 70.0, 3: 70.0, 4: 8.0},  # tail crushed to 8%.
        }
        trainer = FakeTrainer(tables)
        local_weights = {0: _state("local0"), 1: _state("local1"), 2: _state("local2"), 3: _state("local3")}

        per_client, per_class = measure_aggregation_crush(
            trainer,
            _state("pre"),
            local_weights,
            _state("post"),
            idxs_users,
            datanumber_client,
            client_class_counts,
            num_users,
            num_classes,
            tail_class_ratio=0.2,
        )

        # One tail class (id 4), supported only by client 3.
        self.assertEqual(len(per_class), 1)
        row = per_class[0]
        self.assertEqual(row["class_id"], 4)
        self.assertEqual(row["num_support_clients"], 1)
        # Mass share = 54 / 204.
        self.assertAlmostEqual(row["support_mass_share"], 54.0 / 204.0, places=6)
        self.assertAlmostEqual(row["best_local_acc"], 80.0, places=6)
        self.assertAlmostEqual(row["global_post_agg_acc"], 8.0, places=6)
        # The crush: 80% locally, 8% after aggregation.
        self.assertAlmostEqual(row["crush_gap_best"], 72.0, places=6)

        # Model restored to post-aggregation weights for the caller.
        self.assertEqual(trainer.model._current, "post")
        self.assertTrue(trainer.model.training)

        self.assertEqual(len(per_client), 1)
        self.assertEqual(per_client[0]["client_id"], 3)
        self.assertAlmostEqual(per_client[0]["client_local_acc"], 80.0, places=6)

    def test_round_gating(self):
        class Args:
            agg_crush_enable = True
            agg_crush_rounds = "5,20"

        args = Args()
        # epoch is 0-based; rounds are 1-based -> epoch 4 == round 5.
        self.assertTrue(should_log_aggregation_crush(args, epoch=4, is_eval_round=False))
        self.assertTrue(should_log_aggregation_crush(args, epoch=19, is_eval_round=False))
        self.assertFalse(should_log_aggregation_crush(args, epoch=3, is_eval_round=True))

        args.agg_crush_rounds = ""  # empty -> follow eval rounds.
        self.assertTrue(should_log_aggregation_crush(args, epoch=3, is_eval_round=True))
        self.assertFalse(should_log_aggregation_crush(args, epoch=3, is_eval_round=False))

        args.agg_crush_enable = False
        self.assertFalse(should_log_aggregation_crush(args, epoch=4, is_eval_round=True))


if __name__ == "__main__":
    unittest.main()
