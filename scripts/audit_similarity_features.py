"""Audit every cosine-cache row and independently check sampled feed formulas."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.data import load_embeddings
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot
from src.common.similarity import feature_names, KEY, RECENT


def reference_features(row, vectors):
    """Independent Float64 calculation; preserve the original missing-event mask."""
    indices = np.asarray(row['item_id__embedding_idx'], dtype=np.int64)
    history = np.asarray(row['history__item_id__embedding_idx'], dtype=np.int64)
    if ((indices < -1).any() or (indices >= len(vectors)).any()
            or (history < -1).any() or (history >= len(vectors)).any()):
        raise ValueError('raw embedding index outside vector file')
    candidates = np.zeros((len(indices), 128), dtype=np.float64)
    candidates[indices >= 0] = vectors[indices[indices >= 0]]
    candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    user = np.asarray(row['user_embedding'], dtype=np.float64)
    user /= max(float(np.linalg.norm(user)), 1e-12)
    result = {'has_item_embedding': (indices >= 0).astype(float),
              'cos_user': np.clip(candidates @ user, -1., 1.),
              'hist_len': np.full(len(indices), len(history), dtype=float)}
    for n in RECENT:
        recent = history[:n]
        recent = recent[recent >= 0]
        result[f'hist_valid_{n}'] = np.full(len(indices), len(recent), dtype=float)
        if len(recent):
            values = vectors[recent].astype(np.float64)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            cosines = np.clip(candidates @ values.T, -1., 1.)
            result[f'cos_hist_mean_{n}'] = cosines.mean(axis=1)
            result[f'cos_hist_max_{n}'] = cosines.max(axis=1)
        else:
            result[f'cos_hist_mean_{n}'] = np.zeros(len(indices))
            result[f'cos_hist_max_{n}'] = np.zeros(len(indices))
    return result


def audit(raw, prepared, cache, output, *, sample_feeds=3):
    raw, prepared, cache, output = (Path(p) for p in (raw, prepared, cache, output))
    if sample_feeds < 1:
        raise ValueError('sample_feeds must be positive')
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if not source.get('complete') or not manifest.get('complete') or not manifest.get('all_train_parts'):
        raise ValueError('a complete prepared dataset and full cosine cache are required')
    names = feature_names()
    if manifest.get('features') != names or manifest.get('key') != KEY:
        raise ValueError('unexpected cosine feature contract')
    schema = {'split':pl.String,'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
              **{name:pl.Float32 for name in names}}
    started = time.monotonic()
    report = {'complete':False,'scope':'all cached rows: keys, schema, finite values, domains and file fingerprints; independent Float64 formulas on sampled feeds in every shard',
              'sample_feeds_per_shard':sample_feeds,'formula_atol':2e-6,'formula_rtol':2e-6,
              'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'value_signature':manifest.get('value_signature'),'splits':{},'sampled_feeds':0,
              'sampled_candidate_rows':0,'sampled_repeated_item_feeds':0,'bytes':0}
    for split in ('train','val','test'):
        entries = source['splits'][split]['parts']
        group = manifest['splits'][split]
        if (set(group['parts']) != {e['source'] for e in entries} or group['covered_parts'] != len(entries)
                or group['total_parts'] != len(entries)):
            raise ValueError(f'{split}: incomplete or unexpected shard coverage')
        row_count = 0
        for number, entry in enumerate(entries):
            part = group['parts'][entry['source']]
            path = cache / part['path']
            raw_path = raw / split / 'impressions' / entry['source']
            if (fingerprint(path) != part['fingerprint'] or fingerprint(raw_path) != part['source']
                    or part['source'] != entry['source_fingerprint']):
                raise ValueError(f'{split}: file fingerprint mismatch')
            frame = pl.read_parquet(path)
            if frame.schema != schema or frame.height != entry['candidates'] or frame.height != part['rows']:
                raise ValueError(f'{split}: feature schema or row count mismatch')
            keys = [*KEY, 'item_id']
            expected = pl.read_parquet(prepared / entry['candidate_path'], columns=[*keys,'item_id__embedding_idx'])
            if not frame.select(keys).equals(expected.select(keys)) or frame.select(KEY).is_duplicated().any():
                raise ValueError(f'{split}: candidate occurrence keys differ')
            if not np.array_equal(frame['has_item_embedding'].to_numpy(),
                                  (expected['item_id__embedding_idx'].to_numpy() >= 0).astype(np.float32)):
                raise ValueError(f'{split}: candidate availability differs from original indices')
            if frame.null_count().to_numpy().any():
                raise ValueError(f'{split}: null feature or key')
            for name in names:
                values = frame[name].to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f'{split}: nonfinite {name}')
                if name.startswith('cos_') and np.abs(values).max() > 1.000001:
                    raise ValueError(f'{split}: cosine outside [-1,1]: {name}')
                if name == 'has_item_embedding' and not np.isin(values,[0.,1.]).all():
                    raise ValueError(f'{split}: invalid embedding availability')
                if name.startswith('hist_'):
                    if (values < 0).any() or not (values == np.floor(values)).all():
                        raise ValueError(f'{split}: invalid history count: {name}')
                    if name.startswith('hist_valid_') and (values > np.minimum(frame['hist_len'].to_numpy(),int(name.rsplit('_',1)[1]))).any():
                        raise ValueError(f'{split}: history count exceeds original window')
            if frame.filter(pl.col('has_item_embedding') == 0).select(
                    [pl.col(name).abs().max() for name in names if name.startswith('cos_')]).fill_null(0).to_numpy().max() > 1e-7:
                raise ValueError(f'{split}: similarity without candidate embedding')
            original = pl.read_parquet(raw_path, columns=['feed_id','item_id','item_id__embedding_idx',
                                                          'history__item_id__embedding_idx','user_embedding'])
            history = pl.col('history__item_id__embedding_idx')
            history_names = ['hist_len',*[f'hist_valid_{n}' for n in RECENT]]
            counts = original.select('feed_id',history.list.len().alias('expected_hist_len'),
                                     *[history.list.head(n).list.eval(pl.element() >= 0).list.sum().alias(f'expected_hist_valid_{n}')
                                       for n in RECENT]).with_columns(pl.col('feed_id').cast(frame.schema['feed_id']))
            checked = frame.select('feed_id',*history_names).join(counts,on='feed_id',how='left',validate='m:1')
            if checked.null_count().to_numpy().any():
                raise ValueError(f'{split}: history counts do not cover all feeds')
            for name in history_names:
                if not np.array_equal(checked[name].to_numpy(),checked[f'expected_{name}'].to_numpy()):
                    raise ValueError(f'{split}: history counts differ from original events: {name}')
            selected = set(np.linspace(0, original.height-1, min(sample_feeds,original.height),dtype=int).tolist())
            repeated = original['item_id'].list.n_unique() < original['item_id'].list.len()
            selected.update(np.flatnonzero(repeated.to_numpy()).tolist())
            vectors = load_embeddings(raw,split)
            try:
                for index in sorted(selected):
                    row = original.row(index,named=True)
                    actual = frame.filter(pl.col('feed_id') == row['feed_id']).sort('candidate_index')
                    reference = reference_features(row,vectors)
                    if actual['item_id'].to_list() != row['item_id']:
                        raise ValueError(f'{split}: sampled candidate order changed')
                    for name in names:
                        if not np.allclose(actual[name].to_numpy(),reference[name],rtol=2e-6,atol=2e-6):
                            raise ValueError(f'{split}: independent formula mismatch: {name}')
                    report['sampled_feeds'] += 1
                    report['sampled_candidate_rows'] += actual.height
                    report['sampled_repeated_item_feeds'] += int(repeated[index])
            finally:
                vectors._mmap.close()
            row_count += frame.height
            report['bytes'] += path.stat().st_size
            if number == 0 or (number+1) % 32 == 0 or number+1 == len(entries):
                print(f'audit {split}: {number+1}/{len(entries)} shards',flush=True)
        if row_count != source['splits'][split]['candidates']:
            raise ValueError(f'{split}: total candidate count mismatch')
        report['splits'][split] = {'parts':len(entries),'rows':row_count}
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != report['manifest_sha256']:
        raise ValueError('cache manifest changed during audit')
    report.update(complete=True,duration_seconds=round(time.monotonic()-started,2),memory=memory_snapshot(),
                  audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path('competition'))
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/cosine'))
    parser.add_argument('--out',type=Path,default=Path('runs/cosine_cache_audit_full.json'))
    parser.add_argument('--sample-feeds',type=int,default=3)
    args = parser.parse_args()
    audit(args.raw,args.prepared,args.cache,args.out,sample_feeds=args.sample_feeds)
