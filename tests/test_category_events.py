import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from scripts.analyse_history_events import analyse
from src.common.preparation import fingerprint, write_json


class CategoryEventTests(unittest.TestCase):
    def test_fractional_presence_rates_and_domain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, cache = root / 'prepared', root / 'features'
            source = {'complete': True, 'splits': {}}
            manifest = {'complete': True, 'splits': {}}
            for split, targets, shares in (
                ('train', [1, 0, 1], [0.5, 1., 0.]),
                ('val', [0, 1], [0.25, 0.75]),
            ):
                size = len(targets)
                labels = pl.DataFrame({
                    'split': [split] * size, 'feed_id': [0] * size,
                    'candidate_index': list(range(size)), 'item_id': [10] * size, 'target': targets,
                }, schema_overrides={'feed_id': pl.Int32, 'candidate_index': pl.UInt32,
                                     'item_id': pl.Int32, 'target': pl.Int8})
                candidates = Path(split) / 'candidates.parquet'
                feeds = Path(split) / 'feeds.parquet'
                (prepared / split).mkdir(parents=True)
                labels.write_parquet(prepared / candidates)
                pl.DataFrame({'feed_id': [0]}).write_parquet(prepared / feeds)
                entry = {'source': 'part.parquet', 'candidate_path': candidates.as_posix(),
                         'feed_path': feeds.as_posix()}
                source['splits'][split] = {'parts': [entry]}
                features = labels.drop('target').with_columns([
                    pl.Series(f'h_cat_evt{event}', shares if event == 1 else [0.] * size, dtype=pl.Float32)
                    for event in range(1, 11)])
                (cache / split).mkdir(parents=True)
                path = Path(split) / 'part.parquet'
                features.write_parquet(cache / path)
                manifest['splits'][split] = {'parts': {'part.parquet': {
                    'path': path.as_posix(), 'fingerprint': fingerprint(cache / path),
                    'source': fingerprint(prepared / candidates), 'feeds_source': fingerprint(prepared / feeds),
                }}}
            write_json(prepared / 'manifest.json', source)
            manifest['source_manifest_sha256'] = hashlib.sha256((prepared / 'manifest.json').read_bytes()).hexdigest()
            write_json(cache / 'manifest.json', manifest)
            report = analyse(prepared, cache, root / 'report', train_parts=1, entity='category')
            self.assertTrue(report['complete'])
            train = report['splits']['train']['events'][0]
            self.assertEqual((train['rows'], train['positives'], train['feeds']), (2, 1, 1))
            self.assertEqual(train['target_rate'], 0.5)
            self.assertEqual(train['rate_ratio_to_baseline'], 0.75)
            self.assertEqual(train['absent_target_rate'], 1.)
            self.assertNotIn('matching_events', train)
            self.assertIsNone(report['splits']['train']['events'][1]['target_rate'])
            self.assertEqual(report['splits']['val']['events'][0]['target_rate'], 0.5)
            self.assertEqual(report['splits']['val']['events'][0]['candidate_share'], 1.)
            saved = json.loads((root / 'report.json').read_text())
            self.assertEqual(saved['entity'], 'category')
            path = cache / 'train/part.parquet'
            pl.read_parquet(path).with_columns(pl.lit(1.1, dtype=pl.Float32).alias('h_cat_evt1')).write_parquet(path)
            manifest['splits']['train']['parts']['part.parquet']['fingerprint'] = fingerprint(path)
            write_json(cache / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'share exceeds one'):
                analyse(prepared, cache, root / 'invalid', train_parts=1, entity='category')


if __name__ == '__main__':
    unittest.main()
