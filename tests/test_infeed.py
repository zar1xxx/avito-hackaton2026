import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from src.common.infeed import add_infeed_stats, build_feature_part, build_features, feature_names
from src.common.preparation import prepare_dataset, scan_candidates
from src.common.similarity import attach_features


class InfeedTests(unittest.TestCase):
    def frame(self):
        return pl.DataFrame({'split': ['train'] * 6, 'feed_id': [1,1,1,2,2,3],
                             'candidate_index': [0,1,2,0,1,0], 'item_id': [10,10,20,30,40,50],
                             'category_id': [1,1,2,3,3,4],
                             **{f'f{i}': [1.,1.,4.,7.,7.,12.] for i in range(1,6)}})\
            .cast({'feed_id': pl.Int32, 'candidate_index': pl.UInt32, 'item_id': pl.Int32})

    def test_known_ranks_z_scores_and_degenerate_feeds(self):
        result = build_feature_part(self.frame())
        np.testing.assert_allclose(result['f1_rank'], [.5,.5,1.,.75,.75,1.])
        np.testing.assert_allclose(result['f1_z'], [-1/np.sqrt(2),-1/np.sqrt(2),np.sqrt(2),0.,0.,0.], rtol=1e-6)
        np.testing.assert_allclose(result['feed_size'], [3,3,3,2,2,1])
        np.testing.assert_allclose(result['cat_share_feed'], [2/3,2/3,1/3,1.,1.,1.])
        self.assertEqual(result['item_id'].to_list(), self.frame()['item_id'].to_list())
        self.assertTrue(all(result.schema[name] == pl.Float32 for name in feature_names()))

    def test_permutation_invariance_and_no_label_or_position_dependence(self):
        frame = self.frame().with_columns(pl.lit(1).alias('target'), pl.lit(999).alias('pos'))
        baseline = build_feature_part(frame).sort('feed_id', 'candidate_index')
        shuffled = frame.sample(fraction=1., shuffle=True, seed=8).with_columns(pl.lit(0).alias('target'), pl.lit(-1).alias('pos'))
        self.assertTrue(baseline.equals(build_feature_part(shuffled).sort('feed_id','candidate_index')))

    def test_split_isolation_and_invalid_inputs(self):
        first = self.frame().head(3)
        second = first.with_columns(pl.lit('val').alias('split'), *[pl.lit(0.).alias(f'f{i}') for i in range(1,6)])
        result = build_feature_part(pl.concat([first,second]))
        self.assertTrue((result.filter(pl.col('split') == 'val')['f1_z'] == 0).all())
        for frame in (self.frame().head(0), pl.concat([first,first]), self.frame().with_columns(pl.lit(float('nan')).alias('f1'))):
            with self.assertRaises(ValueError):
                build_feature_part(frame)

    def test_shared_reader_uses_finite_population_stats(self):
        from src.common.data import _in_feed_stats
        frame = self.frame().with_columns(pl.col('f1').alias('cos_user'), pl.col('f2').alias('cos_hist_max'))
        legacy = _in_feed_stats(frame)
        expected = add_infeed_stats(frame)
        self.assertTrue(legacy.select(feature_names()).equals(expected.select(feature_names())))
        self.assertEqual(legacy['cos_user_z'][-1],0.)

    def test_full_cache_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, output = root / 'raw', root / 'prepared', root / 'features'
            rng = np.random.default_rng(4)
            for i, split in enumerate(('train','val','test')):
                make_split(raw, split, 6, rng, i*100, split == 'train', split != 'test')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id','item_id').explode('item_id').cast({'feed_id':pl.Int32,'item_id':pl.Int32})\
                .with_columns(pl.lit(0.,dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            prepare_dataset(raw, prepared, root / 'eda')
            manifest = build_features(prepared, output)
            self.assertTrue(manifest['complete'] and manifest['all_train_parts'])
            before = {path:path.stat().st_mtime_ns for path in output.glob('*/*.parquet')}
            build_features(prepared, output, resume=True)
            self.assertEqual(before,{path:path.stat().st_mtime_ns for path in before})
            for split in ('train','val','test'):
                frame = scan_candidates(prepared,split).collect().reverse()
                joined,names = attach_features(frame,output,split,['part-00000.parquet'])
                self.assertEqual(joined.height,frame.height)
                self.assertEqual(names,feature_names())
                self.assertEqual(joined.select(names).null_count().to_numpy().sum(),0)


if __name__ == '__main__':
    unittest.main()
