import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import polars as pl

from src.common.history import _feed_block, build_feature_part, feature_names, build_features
from src.common.preparation import prepare_dataset,write_json,fingerprint
from eval.make_synthetic import make_split
from scripts.audit_history_features import audit
from scripts.analyse_history_events import analyse


class HistoryTests(unittest.TestCase):
    def test_resume_feed_invalidation_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            raw,prepared,features=(root/name for name in ('raw','prepared','features'))
            rng=np.random.default_rng(31)
            for index,split in enumerate(('train','val','test')):
                make_split(raw,split,9,rng,index*100,split=='train',split!='test')
            path=raw/'train/impressions/part-00000.parquet'
            train=pl.read_parquet(path)
            rows=train.to_dicts()
            for column in ('item_id','category_id','item_id__embedding_idx'):
                rows[0][column][1]=rows[0][column][0]
            train=pl.DataFrame(rows,schema=train.schema)
            for index in range(3):
                train.slice(index*3,3).write_parquet(path.with_name(f'part-{index:05d}.parquet'))
            test=pl.read_parquet(raw/'test/impressions/part-00000.parquet')
            test.select('feed_id','item_id').explode('item_id').cast({'feed_id':pl.Int32,'item_id':pl.Int32})\
                .with_columns(pl.lit(0.,dtype=pl.Float32).alias('score')).write_parquet(raw/'sample_submission.parquet')
            source=prepare_dataset(raw,prepared,root/'eda')
            build_features(prepared,features,train_parts=1)
            def interrupt(path,value):
                write_json(path,value)
                if not value['complete'] and value['splits'].get('train',{}).get('covered_parts')==2:
                    raise RuntimeError('interrupted')
            with patch('src.common.history.write_json',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    build_features(prepared,features,train_parts=3,resume=True)
            saved=json.loads((features/'manifest.json').read_text())
            self.assertEqual(saved['splits']['val']['covered_parts'],1)
            self.assertEqual(saved['splits']['test']['covered_parts'],1)
            with patch('src.common.history.build_feature_part',wraps=build_feature_part) as builder:
                build_features(prepared,features,train_parts=3,resume=True)
                self.assertEqual(builder.call_count,1)
            before={path:path.stat().st_mtime_ns for path in features.glob('*/*.parquet')}
            small=build_features(prepared,features,train_parts=1,resume=True)
            self.assertEqual(small['splits']['train']['covered_parts'],3)
            self.assertEqual(before,{path:path.stat().st_mtime_ns for path in before})
            source=json.loads((prepared/'manifest.json').read_text())
            feed_path=prepared/source['splits']['train']['parts'][1]['feed_path']
            pl.read_parquet(feed_path).with_columns(pl.col('history__event_type').list.reverse()).write_parquet(feed_path)
            with patch('src.common.history.build_feature_part',wraps=build_feature_part) as builder:
                build_features(prepared,features,train_parts=3,resume=True)
                self.assertEqual(builder.call_count,1)
            checked=audit(prepared,features,root/'audit.json')
            self.assertTrue(checked['complete'])
            self.assertGreater(checked['sampled_repeated_item_feeds'],0)
            report=analyse(prepared,features,root/'events',train_parts=3)
            for split in ('train','val'):
                expected=[]
                for entry in source['splits'][split]['parts']:
                    feeds=pl.read_parquet(prepared/entry['feed_path'])
                    histories={row['feed_id']:row for row in feeds.iter_rows(named=True)}
                    for candidate in pl.read_parquet(prepared/entry['candidate_path']).iter_rows(named=True):
                        history=histories[candidate['feed_id']]
                        if any(item==candidate['item_id'] and event==1 for item,event in zip(
                                history['history__item_id'],history['history__event_type'])):
                            expected.append(candidate['target'])
                actual=report['splits'][split]['events'][0]
                self.assertEqual(actual['rows'],len(expected))
                self.assertEqual(actual['positives'],sum(expected))
                for event in report['splits'][split]['events']:
                    if event['rows']==0:
                        self.assertIsNone(event['target_rate'])
            corrupt=features/'train/part-00000.parquet'
            data=pl.read_parquet(corrupt).with_columns((pl.col('h_cat_share')+1).alias('h_cat_share'))
            data.write_parquet(corrupt)
            saved=json.loads((features/'manifest.json').read_text())
            saved['splits']['train']['parts']['part-00000.parquet']['fingerprint']=fingerprint(corrupt)
            write_json(features/'manifest.json',saved)
            with self.assertRaisesRegex(ValueError,'independent history formula mismatch'):
                audit(prepared,features,root/'invalid.json')

    def test_hand_computed_block(self):
        # история от новых к старым: товары 7,5,7,9; категории 2,3,2,4; типы 1,2,2,1
        out = _feed_block(np.array([7, 5, 1]), np.array([2, 3, 8]),
                          np.array([7, 5, 7, 9]), np.array([2, 3, 2, 4]), np.array([1, 2, 2, 1]))
        np.testing.assert_allclose(out['h_item_cnt'], [2, 1, 0])
        np.testing.assert_allclose(out['h_item_last'], [0, 1 / 256, 1])
        np.testing.assert_allclose(out['h_item_last_evt'], [1, 2, 0])
        np.testing.assert_allclose(out['h_item_evt1'], [1, 0, 0])
        np.testing.assert_allclose(out['h_item_evt2'], [1, 1, 0])
        np.testing.assert_allclose(out['h_item_evt_nunique'], [2, 1, 0])
        np.testing.assert_allclose(out['h_cat_share'], [0.5, 0.25, 0])
        np.testing.assert_allclose(out['h_cat_is_last'], [1, 0, 0])
        np.testing.assert_allclose(out['h_cat_rank'], [0, 1, 3])
        np.testing.assert_allclose(out['h_cat_evt1'], [0.5, 0, 0])      # тип 1: категории 2 и 4
        np.testing.assert_allclose(out['h_ncats'], [3, 3, 3])
        np.testing.assert_allclose(out['h_user_evt1'], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(out['h_item_cnt_feedrank'], [2.5 / 3, 1.5 / 3, 0.5 / 3])

    def test_empty_history_and_alignment(self):
        cand = pl.DataFrame({'split': ['val'] * 5, 'feed_id': [2, 2, 1, 1, 1], 'candidate_index': [1, 0, 2, 0, 1],
                             'item_id': [10, 11, 5, 6, 7], 'category_id': [1, 1, 2, 3, 2]})
        feeds = pl.DataFrame({'feed_id': [1, 2], 'history__item_id': [[7, 7], []],
                              'history__category_id': [[2, 2], []], 'history__event_type': [[3, 1], []]},
                             schema_overrides={'history__item_id': pl.List(pl.Int64),
                                               'history__category_id': pl.List(pl.Int64),
                                               'history__event_type': pl.List(pl.Int64)})
        out = build_feature_part(cand, feeds)
        self.assertEqual(out.columns[:4], ['split', 'feed_id', 'candidate_index', 'item_id'])
        self.assertEqual(out.columns[4:], feature_names())
        row = out.filter((pl.col('feed_id') == 1) & (pl.col('item_id') == 7))
        self.assertEqual(row['h_item_cnt'][0], 2.0)
        self.assertEqual(row['h_item_last_evt'][0], 3.0)
        empty = out.filter(pl.col('feed_id') == 2)
        self.assertTrue((empty['h_item_cnt'] == 0).all() and (empty['h_item_last'] == 1).all())

    def test_candidate_order_does_not_change_values(self):
        rng = np.random.default_rng(0)
        ci, cc = rng.integers(1, 20, 12), rng.integers(1, 5, 12)
        h = (rng.integers(1, 20, 40), rng.integers(1, 5, 40), rng.integers(1, 11, 40))
        base = _feed_block(ci, cc, *h)
        perm = rng.permutation(12)
        shuffled = _feed_block(ci[perm], cc[perm], *h)
        for name in feature_names():
            np.testing.assert_allclose(shuffled[name], base[name][perm], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
