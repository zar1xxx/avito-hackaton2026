from pathlib import Path
import tempfile
import unittest

from catboost import CatBoostRanker
import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.run_baseline_check import FEATURES, load_validation, pool, run
from src.common.preparation import prepare_dataset
from src.common.validation import make_validation_split
from src.common.embedding_catalogue import build_catalogue
from src.common.similarity import build_features, feature_names


REFERENCE = Path(__file__).resolve().parents[1] / 'competition/baseline/calculate_metrics.py'


class BaselineCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.raw, self.prepared, self.out = root / 'raw', root / 'prepared', root / 'run'
        rng = np.random.default_rng(7)
        for split, count, start, pos, target in [('train', 20, 0, True, True),
                                                 ('val', 10, 100, False, True),
                                                 ('test', 10, 200, False, False)]:
            make_split(self.raw, split, count, rng, start, pos, target)
        test = pl.read_parquet(self.raw / 'test/impressions/part-00000.parquet')
        test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})\
            .with_columns(pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(self.raw / 'sample_submission.parquet')
        prepare_dataset(self.raw, self.prepared, root / 'eda')
        val = pl.read_parquet(self.raw / 'val/impressions/part-00000.parquet')
        self.folds = make_validation_split(val.with_columns(pl.col('feed_id').cast(pl.Int32)))
        self.folds.write_parquet(self.prepared / 'val_folds.parquet')

    @unittest.skipUnless(REFERENCE.is_file(), 'download the supplied evaluator to check a full baseline run')
    def test_end_to_end_and_saved_model_scores(self):
        report = run(self.prepared, self.raw, self.out, iterations=5, depth=3, threads=1, reference_path=REFERENCE)
        self.assertTrue(report['complete'])
        self.assertEqual(report['counts']['selection_feeds'], 3)
        self.assertEqual(report['validation']['pseudo_private']['feeds'], 7)
        self.assertEqual(report['submission']['feeds'], 10)
        self.assertEqual(report['submission']['validator'], 'passed')
        self.assertFalse(report['submission']['uploaded_to_competition'])
        for metrics in report['validation'].values():
            self.assertEqual(metrics['reference_check'], 'passed')
        model = CatBoostRanker()
        model.load_model(str(self.out / 'model.cbm'))
        self.assertEqual(model.feature_names_, FEATURES)
        frame = load_validation(self.prepared)
        saved = pl.read_parquet(self.out / 'validation_predictions.parquet')
        np.testing.assert_array_equal(model.predict(pool(frame, labeled=False)).astype(np.float32), saved['score'].to_numpy())
        self.assertEqual(saved.height, report['counts']['val_rows'])
        with self.assertRaises(FileExistsError):
            run(self.prepared, self.raw, self.out, reference_path=REFERENCE)

    def test_incomplete_val_assignments_rejected(self):
        self.folds.head(9).write_parquet(self.prepared / 'val_folds.parquet')
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            load_validation(self.prepared)

    def test_time_overlap_rejected(self):
        self.folds.with_columns(pl.when(pl.col('feed_id') % 2 == 0).then(pl.lit('public'))
                               .otherwise(pl.lit('private')).alias('leaderboard_split'))\
            .write_parquet(self.prepared / 'val_folds.parquet')
        with self.assertRaisesRegex(ValueError, 'strictly precede'):
            load_validation(self.prepared)

    @unittest.skipUnless(REFERENCE.is_file(), 'download the supplied evaluator to compare cosine features')
    def test_cosine_feature_baseline_end_to_end(self):
        catalogue, features = self.prepared.parent / 'catalogue', self.prepared.parent / 'features'
        build_catalogue(self.raw, catalogue)
        build_features(self.raw, self.prepared, catalogue, features)
        report = run(self.prepared, self.raw, self.out, iterations=3, depth=2, threads=1,
                     reference_path=REFERENCE, feature_root=features)
        self.assertEqual(report['features'], FEATURES + feature_names())
        self.assertEqual(report['submission']['validator'], 'passed')
        self.assertEqual(report['validation']['pseudo_private']['feeds'], 7)


if __name__ == '__main__':
    unittest.main()
