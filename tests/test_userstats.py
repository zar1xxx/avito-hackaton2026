import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from src.common.preparation import write_json,fingerprint
from src.common.similarity import attach_features
from src.common.userstats import build_features,feature_names
from scripts.audit_userstats_features import audit


class UserstatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.prepared=self.root/'prepared'
        entries={
            'train':[
                [(0,7,10,0,0),(0,7,10,1,1),(1,99,40,0,1)],
                [(10,7,10,0,1),(10,7,20,1,0),(11,9,30,1,1)],
                [(20,7,40,0,0),(21,8,20,1,0),(21,8,50,0,1)],
                [(30,8,30,0,1),(31,17,70,1,0)]],
            'val':[[(100,7,10,0,1),(101,99,40,1,0)]],
            'test':[[(200,8,20,0,0),(200,8,50,1,0)]]}
        source={'complete':True,'splits':{}}
        history={10:[1,2,10],11:[1],20:[2],21:[3,3]}
        for split,parts in entries.items():
            saved=[]
            for index,rows in enumerate(parts):
                indices={}
                records=[]
                users={}
                for feed,user,item,pos,target in rows:
                    candidate_index=indices.get(feed,0)
                    indices[feed]=candidate_index+1
                    records.append({'split':split,'feed_id':feed,'candidate_index':candidate_index,
                                    'user_id':user,'item_id':item,'pos':pos,'target':target})
                    users[feed]=user
                frame=pl.DataFrame(records,schema_overrides={'feed_id':pl.Int32,'candidate_index':pl.UInt32,
                    'user_id':pl.Int32,'item_id':pl.Int32,'pos':pl.Int16,'target':pl.Int8})
                if split=='test':
                    frame=frame.drop('pos','target')
                name=f'part-{index:05d}.parquet'
                relative=Path(split)/'candidates'/name
                (self.prepared/relative).parent.mkdir(parents=True,exist_ok=True)
                frame.write_parquet(self.prepared/relative)
                feeds=pl.DataFrame({'feed_id':list(users),'user_id':list(users.values()),
                    'history__event_type':[history.get(feed,[]) for feed in users]},
                    schema_overrides={'history__event_type':pl.List(pl.Int8)})
                feed_path=Path(split)/'feeds'/name
                (self.prepared/feed_path).parent.mkdir(parents=True,exist_ok=True)
                feeds.write_parquet(self.prepared/feed_path)
                saved.append({'source':name,'candidate_path':relative.as_posix(),'feed_path':feed_path.as_posix(),
                              'candidates':len(rows),'feeds':len(users)})
            source['splits'][split]={'parts':saved,'candidates':sum(e['candidates'] for e in saved)}
        write_json(self.prepared/'manifest.json',source)

    def build(self,name):
        return build_features(self.prepared,self.root/name,train_parts=2,memory_mb=256,threads=1)

    def test_user_formulas_and_label_exclusion(self):
        manifest=self.build('features')
        self.assertEqual(len(feature_names()),16)
        self.assertEqual(manifest['donor_summary']['impressions'],6)
        self.assertEqual(manifest['donor_summary']['clicks'],3)
        frame=pl.read_parquet(self.root/'features/train/part-00000.parquet')
        row=frame.row(0,named=True)
        self.assertEqual(row['is_user_impressions'],3)
        self.assertEqual(row['is_user_clicks'],1)
        self.assertAlmostEqual(row['is_user_ctr'],11/23,places=6)
        self.assertAlmostEqual(row['is_user_expected_clicks'],5/3,places=6)
        self.assertAlmostEqual(row['is_user_position_ratio'],33/35,places=6)
        self.assertEqual(row['is_user_history_count'],4)
        self.assertEqual(row['is_user_history_evt1'],1)
        self.assertEqual(row['is_user_history_evt2'],2)
        unknown=frame.row(2,named=True)
        self.assertEqual(unknown['is_user_impressions'],0)
        self.assertEqual(unknown['is_user_history_count'],0)
        self.assertEqual(unknown['is_user_ctr'],.5)
        self.assertEqual(unknown['is_user_position_ratio'],1)
        self.assertTrue(frame.select(feature_names()).row(0)==frame.select(feature_names()).row(1))
        self.assertTrue(audit(self.prepared,self.root/'features',self.root/'audit.json')['complete'])
        path=self.root/'features/train/part-00000.parquet'
        frame.with_columns((pl.col('is_user_history_evt2')+1).alias('is_user_history_evt2')).write_parquet(path)
        manifest['splits']['train']['parts'][path.name]['fingerprint']=fingerprint(path)
        write_json(self.root/'features/manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'independent userstats formula mismatch'):
            audit(self.prepared,self.root/'features',self.root/'invalid.json')
        frame.write_parquet(path)
        manifest['splits']['train']['parts'][path.name]['fingerprint']=fingerprint(path)
        write_json(self.root/'features/manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'overlap'):
            attach_features(frame,self.root/'features','train',['part-00001.parquet'])
        before={p.relative_to(self.root/'features'):pl.read_parquet(p) for p in (self.root/'features').glob('*/*.parquet')}
        source=json.loads((self.prepared/'manifest.json').read_text())
        for split,indices in [('train',[0,3]),('val',[0])]:
            for index in indices:
                path=self.prepared/source['splits'][split]['parts'][index]['candidate_path']
                pl.read_parquet(path).with_columns((1-pl.col('target')).alias('target')).write_parquet(path)
        self.build('flipped')
        for relative,expected in before.items():
            actual=pl.read_parquet(self.root/'flipped'/relative)
            np.testing.assert_array_equal(expected.select(feature_names()).to_numpy().view(np.uint32),
                                          actual.select(feature_names()).to_numpy().view(np.uint32))

    def test_resume_and_zero_positive_donors(self):
        def interrupt(path,value):
            write_json(path,value)
            if value.get('stage')=='user_exposure':
                raise RuntimeError('interrupted')
        with patch('src.common.userstats.write_json',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                self.build('features')
        build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,resume=True)
        before={p:p.stat().st_mtime_ns for p in (self.root/'features').glob('*/*.parquet')}
        def stop_features(path,value):
            write_json(path,value)
            if value.get('stage')=='features_train':
                raise RuntimeError('feature interruption')
        with patch('src.common.userstats.write_json',side_effect=stop_features):
            with self.assertRaisesRegex(RuntimeError,'feature interruption'):
                build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,resume=True)
        partial=json.loads((self.root/'features/manifest.json').read_text())
        self.assertEqual(partial['splits']['train']['covered_parts'],2)
        self.assertEqual(partial['splits']['test']['covered_parts'],1)
        build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,resume=True)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
        source=json.loads((self.prepared/'manifest.json').read_text())
        for index in (1,2):
            path=self.prepared/source['splits']['train']['parts'][index]['candidate_path']
            pl.read_parquet(path).with_columns(pl.lit(0,dtype=pl.Int8).alias('target')).write_parquet(path)
        manifest=self.build('negative')
        self.assertEqual(manifest['donor_summary']['prior_ctr'],0)
        frame=pl.read_parquet(self.root/'negative/val/part-00000.parquet')
        self.assertTrue((frame['is_user_ctr']==0).all())
        self.assertTrue((frame['is_user_position_ratio']==1).all())


if __name__=='__main__':
    unittest.main()
