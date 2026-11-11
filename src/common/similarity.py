"""Batch cosine features using each split's exact vectors and original -1 masks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.embedding_catalogue import EmbeddingCatalogue, catalogue_signature
from src.common.data import load_embeddings
from src.common.preparation import fingerprint, flat_values, write_json, write_parquet_atomic


RECENT = (10, 50)
KEY = ['split', 'feed_id', 'candidate_index']


def feature_names(recent=RECENT):
    return ['has_item_embedding', 'cos_user', 'hist_len',
            *[f'{prefix}_{n}' for n in recent for prefix in ('hist_valid', 'cos_hist_mean', 'cos_hist_max')]]


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def gather(vectors, indices):
    if (indices < -1).any() or (indices >= len(vectors)).any():
        raise ValueError('embedding index outside [-1, vector_rows)')
    valid = indices >= 0
    result = vectors[np.maximum(indices, 0)].astype(np.float32)
    result *= valid[..., None]
    return result


def padded_features(candidate_indices, history_indices, user_vectors, history_lengths, vectors, recent=RECENT):
    """Batched matmul avoids a Python matrix multiplication for every single feed."""
    candidates = unit(gather(vectors, candidate_indices))
    history = unit(gather(vectors, history_indices))
    available = candidate_indices >= 0
    shape = candidate_indices.shape
    result = {
        'has_item_embedding': available.astype(np.float32),
        'cos_user': np.clip(np.einsum('bmd,bd->bm', candidates, unit(user_vectors)), -1., 1.),
        'hist_len': np.broadcast_to(np.asarray(history_lengths, dtype=np.float32)[:, None], shape),
    }
    similarity = np.clip(np.matmul(candidates, history.transpose(0, 2, 1)), -1., 1.)
    for n in recent:
        # Slice the original first N events, then mask absent embeddings. Do not
        # substitute older events for recent events whose embedding is missing.
        valid = history_indices[:, :n] >= 0
        count = valid.sum(axis=1)
        values = similarity[:, :, :n]
        result[f'hist_valid_{n}'] = np.broadcast_to(count.astype(np.float32)[:, None], shape)
        result[f'cos_hist_mean_{n}'] = (values * valid[:, None, :]).sum(axis=2) / np.maximum(count[:, None], 1)
        maximum = np.where(valid[:, None, :], values, -np.inf).max(axis=2)
        result[f'cos_hist_max_{n}'] = np.where(count[:, None] > 0, maximum, 0.)
    for name in feature_names(recent):
        result[name] = np.asarray(result[name], dtype=np.float32)
    return result


def padded_indices(values, offsets, start, end, width):
    lengths = offsets[start + 1:end + 1] - offsets[start:end]
    present = np.arange(width)[None, :] < lengths[:, None]
    if not values.size:
        return np.full((end - start, width), -1, dtype=np.int32), present
    positions = offsets[start:end, None] + np.arange(width)[None, :]
    indices = values[np.minimum(positions, len(values) - 1)]
    return np.where(present, indices, -1).astype(np.int32), present


def build_feature_part(frame, split, vectors, *, batch_size=16, recent=RECENT):
    if batch_size < 1 or not recent or any(n < 1 for n in recent):
        raise ValueError('batch_size and recent lengths must be positive')
    candidate_lengths = frame['item_id'].list.len().to_numpy()
    history_lengths = frame['history__item_id__embedding_idx'].list.len().to_numpy()
    candidate_offsets = np.r_[0, candidate_lengths.cumsum(dtype=np.int64)]
    history_offsets = np.r_[0, history_lengths.cumsum(dtype=np.int64)]
    candidate_values = flat_values(frame['item_id__embedding_idx'])
    history_values = flat_values(frame['history__item_id__embedding_idx'])
    user = flat_values(frame['user_embedding']).reshape(frame.height, 128)
    chunks = {name: [] for name in feature_names(recent)}
    for start in range(0, frame.height, batch_size):
        end = min(start + batch_size, frame.height)
        width = int(candidate_lengths[start:end].max())
        if width < 1:
            raise ValueError('empty candidate list')
        candidate, present = padded_indices(candidate_values, candidate_offsets, start, end, width)
        history, _ = padded_indices(history_values, history_offsets, start, end, max(recent))
        result = padded_features(candidate, history, user[start:end], history_lengths[start:end], vectors, recent)
        for name in chunks:
            chunks[name].append(result[name][present])
    keys = frame.select('feed_id', 'item_id').with_columns(
        pl.int_ranges(0, pl.col('item_id').list.len(), dtype=pl.UInt32).alias('candidate_index'))\
        .explode('item_id', 'candidate_index').with_columns(pl.lit(split).alias('split'))\
        .select(*KEY, 'item_id').cast({'feed_id': pl.Int32, 'item_id': pl.Int32})
    result = keys.with_columns([pl.Series(name, np.concatenate(values), dtype=pl.Float32) for name, values in chunks.items()])
    if result.select(feature_names(recent)).null_count().to_numpy().any():
        raise ValueError('null cosine feature')
    for name in feature_names(recent):
        if not result[name].is_finite().all():
            raise ValueError(f'nonfinite feature: {name}')
    if result.select(KEY).is_duplicated().any():
        raise ValueError('duplicate candidate occurrence key')
    return result


def build_features(raw, prepared, catalogue_root, output, *, train_parts=None, batch_size=16, resume=False):
    raw, prepared, catalogue_root, output = (Path(path).resolve() for path in (raw, prepared, catalogue_root, output))
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError('feature output must be separate from raw data')
    previous = {}
    if (output / 'manifest.json').exists():
        if not resume:
            raise FileExistsError('feature output exists; use --resume')
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    store = EmbeddingCatalogue(catalogue_root)
    if store.raw != raw:
        raise ValueError('catalogue references different raw data')
    catalogue_sha = catalogue_signature(store.manifest)
    implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {'format_version': 1, 'complete': False, 'key': KEY, 'features': feature_names(),
                'recent_events': list(RECENT), 'missing_policy': 'preserve original per-occurrence -1 indices',
                'vector_policy': 'original split version, no cross-split filling',
                'catalogue_sha256': catalogue_sha, 'implementation_sha256': implementation_sha, 'splits': {}}
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    for split in ('train', 'val', 'test'):
        entries = source['splits'][split]['parts']
        if split == 'train' and train_parts is not None:
            if not 1 <= train_parts <= len(entries):
                raise ValueError('train_parts outside source range')
            entries = [entries[int(i)] for i in np.linspace(0, len(entries) - 1, train_parts, dtype=int)]
        saved = {}
        for number, entry in enumerate(entries):
            source_path = raw / split / 'impressions' / entry['source']
            stamp = fingerprint(source_path)
            if stamp != entry['source_fingerprint']:
                raise ValueError('raw shard changed since prepared table generation')
            relative = Path(split) / entry['source']
            path = output / relative
            prior = previous.get('splits', {}).get(split, {}).get('parts', {}).get(entry['source'], {})
            reuse = (resume and previous.get('format_version') == 1
                     and previous.get('implementation_sha256') == implementation_sha
                     and previous.get('catalogue_sha256') == catalogue_sha and prior.get('source') == stamp
                     and path.exists() and prior.get('fingerprint') == fingerprint(path)
                     and pq.read_metadata(path).num_rows == entry['candidates'])
            if not reuse:
                frame = pl.read_parquet(source_path, columns=['feed_id', 'item_id', 'item_id__embedding_idx',
                                                              'history__item_id__embedding_idx', 'user_embedding'])
                # Reopen the file mapping between small feed blocks. This bounds
                # resident mapped pages on Windows even for the 10 GB train file;
                # copied feature columns do not retain views of the memmap.
                chunks = []
                for start in range(0, frame.height, 128):
                    vector_window = load_embeddings(raw, split)
                    try:
                        chunks.append(build_feature_part(frame.slice(start, 128), split, vector_window, batch_size=batch_size))
                    finally:
                        vector_window._mmap.close()
                features = pl.concat(chunks)
                if features.height != entry['candidates']:
                    raise ValueError('feature count does not match prepared candidates')
                write_parquet_atomic(features, path)
                del frame, features
            saved[entry['source']] = {'path': relative.as_posix(), 'rows': entry['candidates'],
                                      'source': stamp, 'fingerprint': fingerprint(path)}
            if number == 0 or (number + 1) % 16 == 0 or number + 1 == len(entries):
                print(f"cosines {split}: {number + 1}/{len(entries)} shards; {time.monotonic() - started:.0f}s", flush=True)
                manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved),
                                             'total_parts': len(source['splits'][split]['parts'])}
                write_json(output / 'manifest.json', manifest)
        manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved),
                                     'total_parts': len(source['splits'][split]['parts'])}
        write_json(output / 'manifest.json', manifest)
    manifest['complete'] = True
    manifest['all_train_parts'] = manifest['splits']['train']['covered_parts'] == manifest['splits']['train']['total_parts']
    manifest['duration_seconds'] = round(time.monotonic() - started, 2)
    write_json(output / 'manifest.json', manifest)
    print(f'Cosine features complete: {output}', flush=True)
    return manifest


def attach_features(frame, feature_root, split, source_parts):
    """Strict one-to-one occurrence join; missing or duplicated rows fail."""
    root = Path(feature_root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('similarity feature set is incomplete')
    missing = set(source_parts) - set(manifest['splits'].get(split, {}).get('parts', {}))
    if missing:
        raise ValueError(f'cosine cache does not cover {len(missing)} requested {split} shards; expand the cache first')
    paths = [root / manifest['splits'][split]['parts'][name]['path'] for name in source_parts]
    features = pl.scan_parquet(paths).collect()
    if features.height != frame.height:
        raise ValueError('similarity feature rows do not exactly cover candidates')
    keys = ['feed_id', 'candidate_index']
    if features['split'].unique().to_list() != [split]:
        raise ValueError('wrong feature split')
    join_keys = [*keys, 'item_id']
    result = frame.join(features.select(*join_keys, *manifest['features']), on=join_keys, how='left', validate='1:1')
    if result.select(manifest['features']).null_count().to_numpy().any():
        raise ValueError('missing similarity feature keys')
    return result, manifest['features']
