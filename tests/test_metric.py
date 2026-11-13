"""Regression checks against the supplied evaluator, including pessimistic ties."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import polars as pl

from src.common.metric import all_variants, mean_ndcg, ndcg_one


REFERENCE = Path(__file__).resolve().parents[1] / "competition/baseline/calculate_metrics.py"


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(ndcg_one([0, 0], [2, 1]), 1.0)
        self.assertEqual(ndcg_one([1, 0], [2, 1]), 1.0)
        self.assertAlmostEqual(ndcg_one([1, 0], [0, 0]), 1 / np.log2(3))
        self.assertEqual(ndcg_one([0] * 30 + [1], list(range(31, 0, -1)), cutoff=30), 0.0)
        self.assertEqual(ndcg_one([1] * 40, [0] * 40, cutoff=30), 1.0)

    def test_equal_feed_weights(self):
        target = [1, 0] + [0] * 100
        self.assertAlmostEqual(mean_ndcg([5, 5] + [2] * 100, target, [0] * 102),
                               (1 / np.log2(3) + 1) / 2)

    def test_invalid_inputs(self):
        for target, score in (([], []), ([1], [0, 1]), ([2], [0]),
                              ([1], [np.nan]), ([0], [np.inf]), ([[1]], [[0]])):
            with self.subTest(target=target, score=score), self.assertRaises(ValueError):
                ndcg_one(target, score)
        for cutoff in (0, -1, 1.5):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                ndcg_one([1], [1], cutoff=cutoff)
        with self.assertRaises(ValueError):
            mean_ndcg([], [], [])
        with self.assertRaises(ValueError):
            mean_ndcg([1], [1, 0], [1, 0])

    @unittest.skipUnless(REFERENCE.is_file(), "download baseline/calculate_metrics.py to compare")
    def test_supplied_evaluator(self):
        spec = importlib.util.spec_from_file_location("reference_metrics", REFERENCE)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = np.random.default_rng(42)
        frames = []
        # Unequal group sizes, no/all positives, tied scores and >30 positives.
        for group, size in enumerate([1, 2, 9, 10, 11, 29, 30, 31, 40, 100] * 10):
            labels = rng.integers(0, 2, size=size)
            if group % 4 == 0:
                labels[:] = 0
            elif group % 4 == 1:
                labels[:] = 1
            scores = rng.integers(-2, 3, size=size).astype(float)
            frames.append(pl.DataFrame({"feed_id": [group * 7] * size,
                                        "target": labels, "score": scores}))
        frame = pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=123)
        for batch in [frame, *frames[:10]]:
            expected = reference.metrics(batch)
            actual = all_variants(batch["feed_id"], batch["target"], batch["score"])
            for local, official in (("ndcg_full", "ndcg"), ("ndcg@10", "ndcg_at_10"),
                                    ("ndcg@30", "ndcg_at_30")):
                with self.subTest(rows=len(batch), metric=official):
                    self.assertAlmostEqual(actual[local], expected[official], delta=1e-10)


if __name__ == "__main__":
    unittest.main()
