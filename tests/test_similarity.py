import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from src.common.data import _feed_features
from src.common.embedding_catalogue import build_catalogue
from src.common.preparation import prepare_dataset, scan_candidates, write_json, fingerprint
from scripts.audit_similarity_features import audit
from src.common.similarity import (attach_features, build_features, build_feature_part,
                                  feature_names, padded_features, required_vector_rows, prefetch_vectors,
                                  LEGACY_IMPLEMENTATIONS, build_prefetched_part)


class SimilarityTests(unittest.TestCase):
    def test_recent_missing_events_are_not_replaced_by_older_events(self):
        vectors = np.zeros((3, 128), dtype=np.int8)
        vectors[0, 0], vectors[1, 1], vectors[2, 0] = 127, 127, -127
        candidates = np.array([[0, 1, -1]], dtype=np.int32)
        history = np.full((1, 50), -1, dtype=np.int32)
        history[0, 10] = 2
        user = vectors[[0]]
        result = padded_features(candidates, history, user, [11], vectors)
        self.assertAlmostEqual(float(result['cos_user'][0, 0]), 1.)
        self.assertEqual(result['cos_hist_max_10'][0, 0], 0.)
        self.assertEqual(result['hist_valid_10'][0, 0], 0.)
        self.assertEqual(result['cos_hist_max_50'][0, 0], -1.)
        self.assertEqual(result['cos_hist_mean_50'][0, 0], -1.)
        self.assertEqual(result['has_item_embedding'][0, 2], 0.)
        self.assertTrue(all(np.isfinite(values).all() for values in result.values()))
        # The older shared reader must use the same original-window semantics.
        row = {'item_id__embedding_idx': [0], 'item_id': [1], 'category_id': [1],
               'user_embedding': vectors[0].tolist(), 'history__item_id': list(range(10, 21)),
               'history__category_id': [1] * 11, 'history__item_id__embedding_idx': [-1] * 10 + [2]}
        old_reader = _feed_features(row, vectors)
        self.assertEqual(old_reader['cos_hist_recent_max'][0], 0.)

    def test_int8_math_converts_before_norms_and_products(self):
        vectors = np.full((2, 128), 127, dtype=np.int8)
        result = padded_features(np.array([[0]]), np.zeros((1, 50), dtype=np.int32), vectors[[1]], [50], vectors)
        for key in ('cos_user', 'cos_hist_mean_10', 'cos_hist_max_50'):
            self.assertAlmostEqual(float(result[key][0, 0]), 1., places=5)

    def test_empty_history_returns_zero_similarity(self):
        vectors = np.ones((1, 128), dtype=np.int8)
        result = padded_features(np.array([[0]]), np.full((1, 50), -1), vectors, [0], vectors)
        self.assertEqual(result['cos_hist_mean_50'][0, 0], 0.)
        self.assertEqual(result['cos_hist_max_50'][0, 0], 0.)

    def test_feature_group_roundtrip_and_strict_key_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, prepared, catalogue, features = (root / name for name in ('raw', 'prepared', 'catalogue', 'features'))
            rng = np.random.default_rng(123)
            for i, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 10, rng, i * 100, split == 'train', split != 'test')
            test = pl.read_parquet(raw / 'test/impressions/part-00000.parquet')
            test.select('feed_id', 'item_id').explode('item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})\
                .with_columns(pl.lit(0., dtype=pl.Float32).alias('score')).write_parquet(raw / 'sample_submission.parquet')
            prepare_dataset(raw, prepared, root / 'eda')
            build_catalogue(raw, catalogue)
            manifest = build_features(raw, prepared, catalogue, features, batch_size=3)
            self.assertTrue(manifest['complete'])
            before = {path: path.stat().st_mtime_ns for path in features.glob('*/*.parquet')}
            build_features(raw, prepared, catalogue, features, resume=True,batch_size=3)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})
            for split in ('train', 'val', 'test'):
                candidates = scan_candidates(prepared, split).collect().reverse()
                joined, names = attach_features(candidates, features, split, ['part-00000.parquet'])
                self.assertEqual(joined.height, candidates.height)
                self.assertEqual(names, feature_names())
                self.assertEqual(joined.select(names).null_count().to_numpy().sum(), 0)
                # Verify vectorized results against independent per-feed cosine math.
                original = pl.read_parquet(raw / split / 'impressions/part-00000.parquet')
                vector = np.memmap(raw / split / 'item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=(2000, 128))
                output = build_feature_part(original, split, vector, batch_size=3)
                row = original.row(0, named=True)
                user = np.array(row['user_embedding'], dtype=np.float64)
                candidate = vector[row['item_id__embedding_idx'][0]].astype(float)
                expected = float(candidate @ user / (np.linalg.norm(candidate) * np.linalg.norm(user)))
                self.assertAlmostEqual(output['cos_user'][0], expected, places=5)
                ids = row['history__item_id__embedding_idx'][:10]
                if ids:
                    hist = vector[ids].astype(float)
                    sims = hist @ candidate / (np.linalg.norm(hist, axis=1) * np.linalg.norm(candidate))
                    self.assertAlmostEqual(output['cos_hist_mean_10'][0], float(sims.mean()), places=5)
                    self.assertAlmostEqual(output['cos_hist_max_10'][0], float(sims.max()), places=5)
                del vector
                with self.assertRaises(ValueError):
                    attach_features(candidates.head(1), features, split, ['part-00000.parquet'])

    def test_sequential_prefetch_is_bitwise_identical_to_mapped_vectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw=Path(temporary)
            make_split(raw,'train',133,np.random.default_rng(88),0,True,True)
            frame=pl.read_parquet(raw/'train/impressions/part-00000.parquet')
            rows=frame.to_dicts()
            rows[0]['item_id__embedding_idx']=[-1]*len(rows[0]['item_id'])
            rows[1]['history__item_id__embedding_idx']=[-1]*len(rows[1]['history__item_id__embedding_idx'])
            frame=pl.DataFrame(rows,schema=frame.schema)
            vector=np.memmap(raw/'train/item_embeddings/data.memmap',dtype=np.int8,mode='r',shape=(2000,128))
            old=pl.concat([build_feature_part(frame.slice(i,128),'train',vector,batch_size=3)
                           for i in range(0,frame.height,128)])
            required=required_vector_rows(frame)
            compact=prefetch_vectors(raw,'train',required,block_rows=17)
            new=build_prefetched_part(frame,'train',compact,required,batch_size=3)
            self.assertTrue(old.select('feed_id','candidate_index','item_id').equals(new.select('feed_id','candidate_index','item_id')))
            for name in feature_names():
                np.testing.assert_array_equal(old[name].to_numpy().view(np.uint32),new[name].to_numpy().view(np.uint32))
            del vector

    def test_cache_expansion_is_not_lost_after_a_smaller_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            raw,prepared,catalogue,features=(root/name for name in ('raw','prepared','catalogue','features'))
            rng=np.random.default_rng(33)
            for i,split in enumerate(('train','val','test')):
                make_split(raw,split,8,rng,i*100,split=='train',split!='test')
            path=raw/'train/impressions/part-00000.parquet'
            frame=pl.read_parquet(path)
            frame.head(4).write_parquet(path)
            frame.tail(4).write_parquet(path.with_name('part-00001.parquet'))
            test=pl.read_parquet(raw/'test/impressions/part-00000.parquet')
            test.select('feed_id','item_id').explode('item_id').cast({'feed_id':pl.Int32,'item_id':pl.Int32})\
                .with_columns(pl.lit(0.,dtype=pl.Float32).alias('score')).write_parquet(raw/'sample_submission.parquet')
            prepare_dataset(raw,prepared,root/'eda')
            build_catalogue(raw,catalogue)
            first=build_features(raw,prepared,catalogue,features,train_parts=1,reader='sequential')
            self.assertEqual(first['splits']['train']['covered_parts'],1)
            first_file=features/'train/part-00000.parquet'
            first_time=first_file.stat().st_mtime_ns
            def checkpoint_then_interrupt(path, value):
                write_json(path, value)
                if not value['complete'] and value['splits'].get('train',{}).get('covered_parts') == 2:
                    raise RuntimeError('simulated interruption')
            with patch('src.common.similarity.write_json',side_effect=checkpoint_then_interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    build_features(raw,prepared,catalogue,features,train_parts=2,resume=True,reader='sequential')
            interrupted=json.loads((features/'manifest.json').read_text())
            self.assertFalse(interrupted['complete'])
            self.assertEqual(interrupted['splits']['train']['covered_parts'],2)
            self.assertEqual(interrupted['splits']['val']['covered_parts'],1)
            self.assertEqual(interrupted['splits']['test']['covered_parts'],1)
            with patch('src.common.similarity.build_prefetched_part',wraps=build_prefetched_part) as builder:
                expanded=build_features(raw,prepared,catalogue,features,train_parts=2,resume=True,reader='sequential')
                self.assertEqual(builder.call_count,0)
            self.assertEqual(expanded['splits']['train']['covered_parts'],2)
            grouped=root/'grouped'
            with patch('src.common.similarity.prefetch_vectors',wraps=prefetch_vectors) as prefetch:
                build_features(raw,prepared,catalogue,grouped,reader='sequential',prefetch_parts=2)
                self.assertEqual(prefetch.call_count,3)  # One shared train read, then val and test.
            for original in features.glob('*/*.parquet'):
                left=pl.read_parquet(original)
                right=pl.read_parquet(grouped/original.relative_to(features))
                self.assertTrue(left.select('split','feed_id','candidate_index','item_id').equals(
                    right.select('split','feed_id','candidate_index','item_id')))
                for name in feature_names():
                    np.testing.assert_array_equal(left[name].to_numpy().view(np.uint32),right[name].to_numpy().view(np.uint32))
            checked=audit(raw,prepared,grouped,root/'audit.json',sample_feeds=2)
            self.assertTrue(checked['complete'])
            # A plausible finite score can satisfy schema and domain checks while
            # still being wrong; the independent formula must reject it.
            corrupt_path=grouped/'train/part-00000.parquet'
            corrupt=pl.read_parquet(corrupt_path)
            scores=corrupt['cos_user'].to_numpy().copy()
            scores[0]=1. if scores[0]!=1. else -1.
            corrupt.with_columns(pl.Series('cos_user',scores)).write_parquet(corrupt_path)
            grouped_manifest=json.loads((grouped/'manifest.json').read_text())
            grouped_manifest['splits']['train']['parts']['part-00000.parquet']['fingerprint']=fingerprint(corrupt_path)
            write_json(grouped/'manifest.json',grouped_manifest)
            with self.assertRaisesRegex(ValueError,'independent formula mismatch: cos_user'):
                audit(raw,prepared,grouped,root/'invalid-audit.json',sample_feeds=2)
            all_times={p:p.stat().st_mtime_ns for p in features.glob('*/*.parquet')}
            smaller=build_features(raw,prepared,catalogue,features,train_parts=1,resume=True)
            self.assertEqual(smaller['splits']['train']['covered_parts'],2)
            self.assertTrue(smaller['all_train_parts'])
            self.assertEqual(first_time,first_file.stat().st_mtime_ns)
            self.assertEqual(all_times,{p:p.stat().st_mtime_ns for p in all_times})

    def test_legacy_cache_is_reused_only_for_its_verified_math(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            raw,prepared,catalogue,features=(root/name for name in ('raw','prepared','catalogue','features'))
            rng=np.random.default_rng(72)
            for i,split in enumerate(('train','val','test')):
                make_split(raw,split,4,rng,i*100,split=='train',split!='test')
            test=pl.read_parquet(raw/'test/impressions/part-00000.parquet')
            test.select('feed_id','item_id').explode('item_id').cast({'feed_id':pl.Int32,'item_id':pl.Int32})\
                .with_columns(pl.lit(0.,dtype=pl.Float32).alias('score')).write_parquet(raw/'sample_submission.parquet')
            prepare_dataset(raw,prepared,root/'eda')
            build_catalogue(raw,catalogue)
            legacy=build_features(raw,prepared,catalogue,features)
            legacy.pop('value_signature')
            legacy['implementation_sha256']=next(iter(LEGACY_IMPLEMENTATIONS))
            manifest_path=features/'manifest.json'
            manifest_path.write_text(json.dumps(legacy),encoding='utf-8')
            with patch('src.common.similarity.build_feature_part',wraps=build_feature_part) as builder:
                build_features(raw,prepared,catalogue,features,resume=True)
                self.assertEqual(builder.call_count,0)
            manifest_path.write_text(json.dumps(legacy),encoding='utf-8')
            with patch('src.common.similarity.value_signature',return_value='changed-math'), \
                 patch('src.common.similarity.build_feature_part',wraps=build_feature_part) as builder:
                build_features(raw,prepared,catalogue,features,resume=True)
                self.assertEqual(builder.call_count,3)


if __name__ == '__main__':
    unittest.main()
