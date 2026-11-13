import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.reproduce_data import prepared_cache_valid, reproduce


class ReproductionTests(unittest.TestCase):
    def test_one_command_outputs_resume_and_corruption_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, eda, catalogue, features = (root / name for name in ('raw', 'prepared', 'eda', 'catalogue', 'features'))
            rng = np.random.default_rng(19)
            for i, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 6, rng, i * 100, split == 'train', split != 'test')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})\
                .with_columns(pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            infeed = root / 'infeed'
            reference = Path('competition/baseline/calculate_metrics.py')
            baseline_out = root / 'baseline' if reference.is_file() else None
            reproduce(raw, prepared, eda, catalogue, features, infeed=infeed, baseline_out=baseline_out,prefetch_parts=2)
            self.assertEqual(json.loads((features/'manifest.json').read_text())['prefetch_parts'],2)
            if baseline_out is not None:
                report = json.loads((baseline_out / 'run.json').read_text(encoding='utf-8'))
                self.assertTrue(report['complete'])
                self.assertEqual(report['submission']['validator'],'passed')
                with self.assertRaises(FileExistsError):
                    reproduce(raw,prepared,eda,catalogue,features,resume=True,infeed=infeed,baseline_out=baseline_out)
            self.assertTrue(prepared_cache_valid(raw, prepared, eda))
            notebook = json.loads((eda / '01_data_eda.ipynb').read_text(encoding='utf-8'))
            self.assertTrue(all(cell['execution_count'] for cell in notebook['cells'] if cell['cell_type'] == 'code'))
            self.assertEqual(json.loads((eda / 'eda_summary.json').read_text(encoding='utf-8'))['splits']['train']['feeds'], 6)
            before = {path: path.stat().st_mtime_ns for path in features.glob('*/*.parquet')}
            reproduce(raw, prepared, eda, catalogue, features, resume=True, infeed=infeed)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
            part = next((prepared / 'train/candidates').glob('*.parquet'))
            part.write_bytes(b'corrupt cache')
            self.assertFalse(prepared_cache_valid(raw, prepared, eda))


if __name__ == '__main__':
    unittest.main()
