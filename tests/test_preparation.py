from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from src.common.preparation import (IdBitmap, distribution, flatten_candidates,
                                    prepare_dataset, scan_candidates, validate_raw, SplitStatistics, flat_values)


class BitmapTests(unittest.TestCase):
    def test_byte_collisions_and_missing_id(self):
        a, b = IdBitmap(), IdBitmap()
        a.update(np.array([0, 1, 2, 2, 3, 7, 8, 50], dtype=np.int32))
        b.update(np.array([0, 2, 7, 8, 99], dtype=np.int32))
        self.assertEqual(a.count(), 6)
        self.assertEqual(b.count(), 4)
        self.assertEqual(a.intersection(b), 3)
        a.update(np.array([2, 9], dtype=np.int32))
        self.assertEqual(a.count(), 7)
        ids = IdBitmap(include_zero=True)
        ids.update(np.array([0, 0, 1], dtype=np.int32))
        self.assertEqual(ids.count(), 2)

    def test_histogram_quantiles(self):
        values = [0, 1, 1, 2, 5, 7, 7, 7, 9]
        result = distribution(Counter(values))
        self.assertAlmostEqual(result['mean'], float(np.mean(values)))
        for name, q in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99)):
            self.assertAlmostEqual(result[name], float(np.quantile(values, q)))

    def test_flat_values_respects_slice_and_empty_lists(self):
        frame = pl.DataFrame({'x': [[1, 2], [], [3, 4], [5]]})
        self.assertEqual(flat_values(frame.slice(2, 1)['x']).tolist(), [3, 4])
        self.assertEqual(flat_values(frame.slice(1, 1)['x']).tolist(), [])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.raw, self.out, self.eda = root / 'raw', root / 'prepared', root / 'eda'
        rng = np.random.default_rng(3)
        for split, offset, targets, pos in [('train', 0, True, True), ('val', 100, True, False), ('test', 200, False, False)]:
            make_split(self.raw, split, 6, rng, offset, pos, targets)
        self.train = self.raw / 'train/impressions/part-00000.parquet'
        test = pl.read_parquet(self.raw / 'test/impressions/part-00000.parquet')
        test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})\
            .with_columns(pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(self.raw / 'sample_submission.parquet')

    def test_flatten_preserves_alignment(self):
        frame = pl.read_parquet(self.train)
        flat = flatten_candidates(frame, 'train')
        self.assertEqual(flat.height, frame['item_id'].list.len().sum())
        self.assertEqual(flat['item_id'].to_list(), [item for row in frame['item_id'] for item in row])
        self.assertEqual(flat['target'].to_list(), [value for row in frame['target'] for value in row])
        self.assertEqual(flat.schema['feed_id'], pl.Int32)
        self.assertEqual(flat.schema['f1'], pl.Float32)
        self.assertNotIn('user_embedding', flat.columns)
        self.assertNotIn('history__item_id', flat.columns)
        self.assertEqual(flat['candidate_index'].to_list(), [i for row in frame['item_id'] for i in range(len(row))])

    def test_full_pipeline_and_resume(self):
        summary = prepare_dataset(self.raw, self.out, self.eda)
        self.assertTrue(summary['complete'])
        for split in ('train', 'val', 'test'):
            prepared = scan_candidates(self.out, split).collect()
            self.assertEqual(prepared.height, summary['splits'][split]['candidates'])
            self.assertEqual(summary['splits'][split]['feeds'], 6)
            raw = pl.read_parquet(self.raw / split / 'impressions/part-00000.parquet')
            items = {item for row in raw['item_id'] for item in row if item > 0}
            self.assertEqual(summary['splits'][split]['unique_candidate_items'], len(items))
            if split != 'test':
                expected_positive = sum(sum(row) for row in raw['target'])
                self.assertEqual(summary['splits'][split]['positives'], expected_positive)
                self.assertAlmostEqual(summary['splits'][split]['ctr'], expected_positive / prepared.height)
        train_items = set(scan_candidates(self.out, 'train').collect()['item_id'].to_list())
        val_items = set(scan_candidates(self.out, 'val').collect()['item_id'].to_list())
        overlap = next(row for row in summary['overlaps'] if row['left'] == 'train' and row['right'] == 'val' and row['entity'] == 'candidate_items')
        self.assertEqual(overlap['intersection'], len((train_items & val_items) - {0}))
        before = {path: path.stat().st_mtime_ns for path in self.out.glob('*/candidates/*.parquet')}
        resumed = prepare_dataset(self.raw, self.out, self.eda, resume=True)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
        self.assertEqual(summary['splits'], resumed['splits'])
        self.assertEqual(summary['overlaps'], resumed['overlaps'])
        # Damaged output is regenerated rather than trusted merely because it exists.
        damaged = next(iter(before))
        damaged.write_bytes(b'bad parquet')
        prepare_dataset(self.raw, self.out, self.eda, resume=True)
        self.assertGreater(pl.read_parquet(damaged).height, 0)
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.raw, self.out, self.eda)

    def test_bad_index_rejected(self):
        frame = pl.read_parquet(self.train).with_columns(pl.col('item_id__embedding_idx').list.eval(pl.lit(-2)))
        with self.assertRaises(ValueError):
            validate_raw(frame, 'train', 2000)

    def test_empty_history_and_missing_embedding(self):
        frame = pl.read_parquet(self.train)
        rows = frame.to_dicts()
        for row in rows:
            row['item_id__embedding_idx'][0] = -1
            for name in row:
                if name.startswith('history__'):
                    row[name] = []
        frame = pl.DataFrame(rows, schema=frame.schema)
        validate_raw(frame, 'train', 2000)
        stats = SplitStatistics('train')
        stats.update(frame)
        summary = stats.summary()
        self.assertEqual(summary['candidate_missing_embeddings'], frame.height)
        self.assertEqual(summary['history_events'], 0)
        self.assertEqual(summary['unique_history_items'], 0)
        self.assertEqual(summary['history_missing_embedding_rate'], 0.)

    def test_test_key_mismatch_rejected(self):
        template = self.raw / 'sample_submission.parquet'
        pl.read_parquet(template).with_columns((pl.col('item_id') + 1).alias('item_id')).write_parquet(template)
        with self.assertRaisesRegex(ValueError, 'sample_submission'):
            prepare_dataset(self.raw, self.out, self.eda)
        self.assertFalse(json.loads((self.out / 'manifest.json').read_text())['complete'])

    def test_repeated_train_item_preserved_with_unique_row_key(self):
        frame = pl.read_parquet(self.train)
        rows = frame.to_dicts()
        rows[0]['item_id'][1] = rows[0]['item_id'][0]
        rows[0]['target'][0], rows[0]['target'][1] = 0, 1
        pl.DataFrame(rows, schema=frame.schema).write_parquet(self.train)
        summary = prepare_dataset(self.raw, self.out, self.eda)
        prepared = scan_candidates(self.out, 'train').collect()
        self.assertTrue(prepared.select('feed_id', 'item_id').is_duplicated().any())
        self.assertFalse(prepared.select('split', 'feed_id', 'candidate_index').is_duplicated().any())
        self.assertEqual(prepared.height, frame['item_id'].list.len().sum())
        self.assertEqual(summary['splits']['train']['duplicate_item_pairs'], 1)
        self.assertEqual(summary['splits']['train']['conflicting_target_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
