import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from src.common.itemstats import build_features,feature_names
from src.common.preparation import write_json
from src.common.similarity import attach_features


class ItemstatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.prepared=self.root/'prepared'
        source={'complete':True,'splits':{}}
        entries={
            'train':[[ (10,1,0,0),(10,1,1,1),(20,2,2,1),(99,9,3,0)],
                     [(10,1,0,1),(10,1,1,0),(20,2,1,1)],
                     [(10,1,0,0),(30,2,1,0),(20,2,0,1)],
                     [(30,2,0,1),(40,7,1,1)]],
            'val':[[(10,1,0,1),(10,1,1,0),(99,9,2,0),(40,7,3,1)]],
            'test':[[(40,7,0,0),(20,2,1,0)]]}
        for split,parts in entries.items():
            saved=[]
            for index,rows in enumerate(parts):
                frame=pl.DataFrame({'split':[split]*len(rows),'feed_id':[index]*len(rows),
                    'candidate_index':list(range(len(rows))),'item_id':[r[0] for r in rows],
                    'category_id':[r[1] for r in rows],'pos':[r[2] for r in rows],'target':[r[3] for r in rows]},
                    schema_overrides={'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
                                      'category_id':pl.Int8,'pos':pl.Int16,'target':pl.Int8})
                if split=='test':
                    frame=frame.drop('target','pos')
                name=f'part-{index:05d}.parquet'
                relative=Path(split)/'candidates'/name
                (self.prepared/relative).parent.mkdir(parents=True,exist_ok=True)
                frame.write_parquet(self.prepared/relative)
                items,categories,events=([10,10,99],[1,1,9],[1,2,10]) if index==1 else ([10,20],[1,2],[2,1])
                feeds=pl.DataFrame({'feed_id':[index],'history__item_id':[items],
                    'history__category_id':[categories],'history__event_type':[events]})
                feed_relative=Path(split)/'feeds'/name
                (self.prepared/feed_relative).parent.mkdir(parents=True,exist_ok=True)
                feeds.write_parquet(self.prepared/feed_relative)
                saved.append({'source':name,'candidate_path':relative.as_posix(),'feed_path':feed_relative.as_posix(),
                              'candidates':len(rows),'feeds':1})
            source['splits'][split]={'parts':saved,'candidates':sum(p['candidates'] for p in saved)}
        write_json(self.prepared/'manifest.json',source)

    def build(self,name):
        return build_features(self.prepared,self.root/name,train_parts=2,memory_mb=256,threads=1)

    def test_formulas_exclusion_and_unseen_values(self):
        manifest=self.build('features')
        self.assertEqual(manifest['donor_summary']['impressions'],6)
        self.assertEqual(manifest['donor_summary']['clicks'],3)
        self.assertTrue(set(manifest['excluded_train_sources']).isdisjoint(manifest['identity']['donor_sources']))
        frame=pl.read_parquet(self.root/'features/train/part-00000.parquet')
        self.assertEqual(frame.height,4)
        self.assertEqual(len(feature_names()),32)
        row=frame.row(0,named=True)
        self.assertEqual(row['is_item_impressions'],3)
        self.assertEqual(row['is_item_clicks'],1)
        self.assertAlmostEqual(row['is_item_expected_clicks'],5/3,places=6)
        self.assertAlmostEqual(row['is_item_ctr'],11/23,places=6)
        self.assertAlmostEqual(row['is_item_position_ratio'],33/35,places=6)
        self.assertEqual(row['is_item_history_count'],3)
        self.assertEqual(row['is_item_history_evt1'],1)
        self.assertEqual(row['is_item_history_evt2'],2)
        unknown=frame.filter(pl.col('item_id')==99).row(0,named=True)
        self.assertEqual(unknown['is_item_impressions'],0)
        self.assertEqual(unknown['is_item_ctr'],.5)
        self.assertEqual(unknown['is_item_position_ratio'],1)
        self.assertEqual(unknown['is_item_history_evt10'],1)
        with self.assertRaisesRegex(ValueError,'overlap'):
            attach_features(frame,self.root/'features','train',['part-00001.parquet'])
        baseline={p.relative_to(self.root/'features'):pl.read_parquet(p) for p in (self.root/'features').glob('*/*.parquet')}
        source=json.loads((self.prepared/'manifest.json').read_text())
        for split,indices in [('train',[0,3]),('val',[0])]:
            for index in indices:
                path=self.prepared/source['splits'][split]['parts'][index]['candidate_path']
                pl.read_parquet(path).with_columns((1-pl.col('target')).alias('target')).write_parquet(path)
        self.build('changed_labels')
        for relative,expected in baseline.items():
            actual=pl.read_parquet(self.root/'changed_labels'/relative)
            np.testing.assert_array_equal(expected.select(feature_names()).to_numpy().view(np.uint32),
                                          actual.select(feature_names()).to_numpy().view(np.uint32))
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            build_features(self.prepared,self.root/'features',train_parts=1,resume=True)

    def test_interrupted_aggregation_and_parquet_resume(self):
        def interrupt(path,value):
            write_json(path,value)
            if value.get('stage')=='item_exposure':
                raise RuntimeError('interrupted')
        with patch('src.common.itemstats.write_json',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                self.build('features')
        self.assertFalse(json.loads((self.root/'features/manifest.json').read_text())['complete'])
        manifest=build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        self.assertTrue(manifest['complete'])
        before={p:p.stat().st_mtime_ns for p in (self.root/'features').glob('*/*.parquet')}
        def interrupt_features(path,value):
            write_json(path,value)
            if value.get('stage')=='features_train':
                raise RuntimeError('feature interruption')
        with patch('src.common.itemstats.write_json',side_effect=interrupt_features):
            with self.assertRaisesRegex(RuntimeError,'feature interruption'):
                build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        saved=json.loads((self.root/'features/manifest.json').read_text())
        self.assertEqual(saved['splits']['train']['covered_parts'],2)
        self.assertEqual(saved['splits']['val']['covered_parts'],1)
        build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})


if __name__=='__main__':
    unittest.main()
