import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from src.common.data import _feed_features
from src.common.embedding_catalogue import build_catalogue
from src.common.preparation import prepare_dataset, scan_candidates
from src.common.similarity import (attach_features, build_features, build_feature_part,
                                  feature_names, padded_features)


class SimilarityTests(unittest.TestCase):
    def test_recent_missing_events_are_not_replaced_by_older_events(self):
        vectors = np.zeros((3, 128), dtype=np.int8)
        vectors[0, 0], vectors[1, 1], vectors[2, 0] = 127, 127, -127
        candidates = np.array([[0, 1, -1]], dtype=np.int32)
        history = np.full((1, 50), -1, dtype=np.int32)
        history[0, 10] = 2
        user = vectors[[0]]
        result = padded_features(candidates, history, user, [11], vectors)
        self.assertAlmostEqual(float(result['cos_user'][0, 0]), 1.)
        self.assertEqual(result['cos_hist_max_10'][0, 0], 0.)
        self.assertEqual(result['hist_valid_10'][0, 0], 0.)
        self.assertEqual(result['cos_hist_max_50'][0, 0], -1.)
        self.assertEqual(result['cos_hist_mean_50'][0, 0], -1.)
        self.assertEqual(result['has_item_embedding'][0, 2], 0.)
        self.assertTrue(all(np.isfinite(values).all() for values in result.values()))
        # The older shared reader must use the same original-window semantics.
        row = {'item_id__embedding_idx': [0], 'item_id': [1], 'category_id': [1],
               'user_embedding': vectors[0].tolist(), 'history__item_id': list(range(10, 21)),
               'history__category_id': [1] * 11, 'history__item_id__embedding_idx': [-1] * 10 + [2]}
        old_reader = _feed_features(row, vectors)
        self.assertEqual(old_reader['cos_hist_recent_max'][0], 0.)

    def test_int8_math_converts_before_norms_and_products(self):
        vectors = np.full((2, 128), 127, dtype=np.int8)
        result = padded_features(np.array([[0]]), np.zeros((1, 50), dtype=np.int32), vectors[[1]], [50], vectors)
        for key in ('cos_user', 'cos_hist_mean_10', 'cos_hist_max_50'):
            self.assertAlmostEqual(float(result[key][0, 0]), 1., places=5)

    def test_empty_history_returns_zero_similarity(self):
        vectors = np.ones((1, 128), dtype=np.int8)
        result = padded_features(np.array([[0]]), np.full((1, 50), -1), vectors, [0], vectors)
        self.assertEqual(result['cos_hist_mean_50'][0, 0], 0.)
        self.assertEqual(result['cos_hist_max_50'][0, 0], 0.)

    def test_feature_group_roundtrip_and_strict_key_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, catalogue, features = (root / name for name in ('raw', 'prepared', 'catalogue', 'features'))
            rng = np.random.default_rng(123)
            for i, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 10, rng, i * 100, split == 'train', split != 'test')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})\
                .with_columns(pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            prepare_dataset(raw, prepared, root / 'eda')
            build_catalogue(raw, catalogue)
            manifest = build_features(raw, prepared, catalogue, features, batch_size=3)
            self.assertTrue(manifest['complete'])
            before = {path: path.stat().st_mtime_ns for path in features.glob('*/*.parquet')}
            build_features(raw, prepared, catalogue, features, resume=True)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
            for split in ('train', 'val', 'test'):
                candidates = scan_candidates(prepared, split).collect().reverse()
                joined, names = attach_features(candidates, features, split, ['part-00000.parquet'])
                self.assertEqual(joined.height, candidates.height)
                self.assertEqual(names, feature_names())
                self.assertEqual(joined.select(names).null_count().to_numpy().sum(), 0)
                # Verify vectorized results against independent per-feed cosine math.
                original = pl.read_parquet(raw / split / 'impressions/part-00000.parquet')
                vector = np.memmap(raw / split / 'item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=(2000, 128))
                output = build_feature_part(original, split, vector, batch_size=3)
                row = original.row(0, named=True)
                user = np.array(row['user_embedding'], dtype=np.float64)
                candidate = vector[row['item_id__embedding_idx'][0]].astype(float)
                expected = float(candidate @ user / (np.linalg.norm(candidate) * np.linalg.norm(user)))
                self.assertAlmostEqual(output['cos_user'][0], expected, places=5)
                ids = row['history__item_id__embedding_idx'][:10]
                if ids:
                    hist = vector[ids].astype(float)
                    sims = hist @ candidate / (np.linalg.norm(hist, axis=1) * np.linalg.norm(candidate))
                    self.assertAlmostEqual(output['cos_hist_mean_10'][0], float(sims.mean()), places=5)
                    self.assertAlmostEqual(output['cos_hist_max_10'][0], float(sims.max()), places=5)
                del vector
                with self.assertRaises(ValueError):
                    attach_features(candidates.head(1), features, split, ['part-00000.parquet'])


if __name__ == '__main__':
    unittest.main()
