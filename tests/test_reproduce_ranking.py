import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.reproduce_ranking import reproduce_ranking


class RankingReproductionTests(unittest.TestCase):
    @unittest.skipUnless(Path('competition/baseline/calculate_metrics.py').is_file(), 'supplied evaluator required')
    def test_all_groups_submission_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            rng = np.random.default_rng(23)
            for index, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 6, rng, index * 100, split == 'train', split != 'test')
            impressions = raw / 'train/impressions'
            donor = pl.read_parquet(impressions / 'part-00000.parquet').with_columns(
                (pl.col('feed_id') + 1000).alias('feed_id'))
            donor.write_parquet(impressions / 'part-00001.parquet')
            with self.assertRaisesRegex(ValueError, 'donor shard'):
                reproduce_ranking(raw, root / 'invalid', prepared=root / 'unprepared',
                                  train_parts=2, itemstats=True)
            self.assertFalse((root / 'unprepared').exists())
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast(
                {'feed_id': pl.Int32, 'item_id': pl.Int32}).with_columns(
                    pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            options = dict(prepared=root / 'prepared', feature_dir=root / 'features', catalogue=root / 'catalogue',
                           train_parts=1, infeed=True, history=True, itemstats=True, userstats=True,
                           iterations=2, depth=2, threads=1, memory_mb=256)
            first = reproduce_ranking(raw, root / 'first', **options)
            self.assertTrue(first['complete'])
            self.assertEqual(len(first['features']), 121)
            self.assertEqual(first['submission']['validator'], 'passed')
            for group in ('itemstats', 'userstats'):
                manifest = json.loads((root / f'features/{group}-001/manifest.json').read_text())
                self.assertEqual(manifest['excluded_train_sources'], ['part-00000.parquet'])
                self.assertEqual(manifest['identity']['donor_sources'], ['part-00001.parquet'])
            for name, columns in (
                ('val_pred.parquet', ['feed_id', 'item_id', 'score', 'target']),
                ('test_pred.parquet', ['feed_id', 'item_id', 'score']),
            ):
                frame = pl.read_parquet(root / 'first' / name)
                self.assertEqual(frame.columns, columns)
                self.assertTrue(frame['score'].is_finite().all())
            pipeline = json.loads((root / 'first/pipeline.json').read_text())
            self.assertTrue(pipeline['complete'])
            timestamps = {path: path.stat().st_mtime_ns for path in (root / 'features').glob('*/*/*.parquet')}
            second = reproduce_ranking(raw, root / 'second', resume=True, **options)
            self.assertTrue(second['complete'])
            self.assertEqual(timestamps, {path: path.stat().st_mtime_ns for path in timestamps})
            self.assertTrue(pl.read_parquet(root / 'first/test_pred.parquet').equals(
                pl.read_parquet(root / 'second/test_pred.parquet')))
            with self.assertRaises(FileExistsError):
                reproduce_ranking(raw, root / 'first', resume=True, **options)


if __name__ == '__main__':
    unittest.main()
