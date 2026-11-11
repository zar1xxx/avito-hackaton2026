import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from scripts.compare_baselines import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.a, self.b, self.output = root / 'a', root / 'b', root / 'comparison.json'
        base = 1 / np.log2(3)
        for path, score, metric in ((self.a, [0., 1.], base), (self.b, [1., 0.], 1.)):
            path.mkdir()
            data = pl.DataFrame({'feed_id': [1,1,2,2,3,3,4,4], 'candidate_index': [0,1]*4,
                                 'item_id': list(range(8)), 'target': [1,0]*4,
                                 'leaderboard_split': ['public']*4 + ['private']*4, 'score': score*4})
            data.write_parquet(path / 'validation_predictions.parquet')
            report = {'complete': True, 'parameters': {'seed': 42}, 'prepared_manifest_sha256': 'same', 'train_parts': ['one'],
                      'validation': {name: {'ndcg@30': metric} for name in ('all', 'pseudo_public', 'pseudo_private')}}
            (path / 'run.json').write_text(json.dumps(report), encoding='utf-8')

    def test_paired_gain(self):
        result = compare(self.a, self.b, self.output, replicates=100)
        expected = 1 - 1 / np.log2(3)
        self.assertAlmostEqual(result['metrics']['pseudo_private']['delta'], expected)
        np.testing.assert_allclose(result['paired_bootstrap_private']['delta_95_ci'], [expected, expected])

    def test_changed_hyperparameters_rejected(self):
        path = self.b / 'run.json'
        report = json.loads(path.read_text())
        report['parameters']['seed'] = 99
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'parameters'):
            compare(self.a, self.b, self.output)

    def test_changed_holdout_rejected(self):
        path = self.b / 'validation_predictions.parquet'
        pl.read_parquet(path).with_columns(pl.lit('private').alias('leaderboard_split')).write_parquet(path)
        with self.assertRaisesRegex(ValueError, 'assignments'):
            compare(self.a, self.b, self.output)
