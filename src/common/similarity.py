"""Batch cosine features using each split's exact vectors and original -1 masks."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.embedding_catalogue import EmbeddingCatalogue, catalogue_signature
from src.common.data import load_embeddings
from src.common.preparation import fingerprint, flat_values, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot


RECENT = (10, 50)
FEED_WINDOW = 128
KEY = ['split', 'feed_id', 'candidate_index']
# Published mapped-reader implementation whose cached values use the same math.
LEGACY_IMPLEMENTATIONS = {
    '4efada00a34d6f30538a46b1ee0a24e2db416f333348ff6c3d70a467ae7e7dae':
        '2677b517a4025814566c7f53d30f525692f2582aed7149bebaee0c505f0818b7',
}


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
    if not len(vectors):
        return np.zeros(indices.shape + (128,), dtype=np.float32)
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


def remap_indices(indices, vector_rows):
    if (indices < -1).any():
        raise ValueError('embedding index below -1')
    result = np.full(indices.shape, -1, dtype=np.int32)
    valid = indices >= 0
    if valid.any():
        positions = np.searchsorted(vector_rows, indices[valid])
        if (positions >= len(vector_rows)).any() or not np.array_equal(vector_rows[positions], indices[valid]):
            raise ValueError('prefetched vectors do not cover all requested indices')
        result[valid] = positions
    return result


def required_vector_rows(frame, recent=RECENT):
    candidates = flat_values(frame['item_id__embedding_idx'])
    history = flat_values(frame['history__item_id__embedding_idx'])
    lengths = frame['history__item_id__embedding_idx'].list.len().to_numpy()
    offsets = np.r_[0, lengths.cumsum(dtype=np.int64)]
    recent_indices, _ = padded_indices(history, offsets, 0, frame.height, max(recent))
    values = np.concatenate([candidates[candidates >= 0], recent_indices[recent_indices >= 0]])
    return np.unique(values)


def prefetch_vectors(raw, split, rows, *, block_rows=65536):
    """Read selected vectors in file order, retaining only the needed int8 rows."""
    directory = Path(raw) / split / 'item_embeddings'
    meta = json.loads((directory / 'data.meta').read_text(encoding='utf-8'))
    if meta['order'] != 'C' or meta['dtype'] != 'int8' or meta['shape'][1] != 128:
        raise ValueError('sequential prefetch requires C-order int8 vectors of dimension 128')
    if block_rows < 1 or (len(rows) and (rows[0] < 0 or rows[-1] >= meta['shape'][0])):
        raise ValueError('invalid prefetch range')
    if len(rows) > 1 and not (np.diff(rows) > 0).all():
        raise ValueError('prefetch rows must be sorted and unique')
    output = np.empty((len(rows),128),dtype=np.int8)
    buffer = np.empty((block_rows,128),dtype=np.int8)
    with (directory / 'data.memmap').open('rb') as source:
        for block in np.unique(rows // block_rows):
            first = int(block) * block_rows
            start,end = np.searchsorted(rows,[first,first+block_rows])
            source.seek(first*128)
            size = source.readinto(buffer)
            if size < min(block_rows,meta['shape'][0]-first)*128:
                raise ValueError('truncated vector file')
            output[start:end] = buffer[rows[start:end]-first]
    return output


def value_signature(batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    # Reader strategy and unrelated cache/CLI changes do not change numerical values.
    functions = (feature_names, unit, gather, padded_features, padded_indices)
    content = '\n'.join(inspect.getsource(function) for function in functions)
    # Invalidate non-divisor batches produced before reader window alignment.
    assembly_version = 1 if FEED_WINDOW % batch_size == 0 else 2
    content += json.dumps({'assembly_version':assembly_version,'recent':RECENT,'batch_size':batch_size,'feed_window':FEED_WINDOW})
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def build_feature_part(frame, split, vectors, *, batch_size=16, recent=RECENT, vector_rows=None):
    if batch_size < 1 or not recent or any(n < 1 for n in recent):
        raise ValueError('batch_size and recent lengths must be positive')
    candidate_lengths = frame['item_id'].list.len().to_numpy()
    history_lengths = frame['history__item_id__embedding_idx'].list.len().to_numpy()
    candidate_offsets = np.r_[0, candidate_lengths.cumsum(dtype=np.int64)]
    history_offsets = np.r_[0, history_lengths.cumsum(dtype=np.int64)]
    candidate_values = flat_values(frame['item_id__embedding_idx'])
    if vector_rows is not None:
        candidate_values = remap_indices(candidate_values,vector_rows)
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
        if vector_rows is not None:
            history = remap_indices(history,vector_rows)
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


def build_prefetched_part(frame, split, vectors, vector_rows, *, batch_size=16):
    # Match mapped-reader batch boundaries even when batch_size does not divide
    # the feed window; padding shape can affect the last bits of matmul results.
    return pl.concat([
        build_feature_part(frame.slice(start, FEED_WINDOW), split, vectors,
                           batch_size=batch_size, vector_rows=vector_rows)
        for start in range(0, frame.height, FEED_WINDOW)
    ])


def build_features(raw, prepared, catalogue_root, output, *, train_parts=None, batch_size=16, resume=False, reader='auto', prefetch_parts=1):
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
    if not 1 <= prefetch_parts <= 16:
        raise ValueError('prefetch_parts must be between 1 and 16')
    if train_parts is not None and not 1 <= train_parts <= len(source['splits']['train']['parts']):
        raise ValueError('train_parts outside source range')
    store = EmbeddingCatalogue(catalogue_root)
    if store.raw != raw:
        raise ValueError('catalogue references different raw data')
    catalogue_sha = catalogue_signature(store.manifest)
    implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    numerical_signature = value_signature(batch_size)
    if reader not in ('auto','mapped','sequential'):
        raise ValueError('reader must be auto, mapped or sequential')
    compatible = (previous.get('format_version') == 1 and
                  (previous.get('value_signature') == numerical_signature or
                   LEGACY_IMPLEMENTATIONS.get(previous.get('implementation_sha256')) == numerical_signature))
    manifest = {'format_version': 1, 'complete': False, 'key': KEY, 'features': feature_names(),
                'recent_events': list(RECENT), 'missing_policy': 'preserve original per-occurrence -1 indices',
                'vector_policy': 'original split version, no cross-split filling',
                'catalogue_sha256': catalogue_sha, 'implementation_sha256': implementation_sha,
                'value_signature':numerical_signature,'reader':reader,'batch_size':batch_size,
                'prefetch_parts':prefetch_parts,'splits': {}}
    # Keep resumable records for later splits while an earlier split is running.
    if compatible and previous.get('catalogue_sha256') == catalogue_sha:
        manifest['splits'] = json.loads(json.dumps(previous.get('splits', {})))
        for group in manifest['splits'].values():
            for prior in group['parts'].values():
                prior.setdefault('producer_implementation_sha256', previous.get('implementation_sha256'))
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    for split in ('train', 'val', 'test'):
        all_entries = source['splits'][split]['parts']
        entries = all_entries
        if split == 'train' and train_parts is not None:
            if not 1 <= train_parts <= len(entries):
                raise ValueError('train_parts outside source range')
            entries = [entries[int(i)] for i in np.linspace(0, len(entries) - 1, train_parts, dtype=int)]
        requested = {entry['source'] for entry in entries}
        old_parts = previous.get('splits',{}).get(split,{}).get('parts',{})
        def reusable(entry):
            prior = old_parts.get(entry['source'],{})
            path = output / split / entry['source']
            return (resume and compatible and previous.get('catalogue_sha256') == catalogue_sha
                    and prior.get('source') == fingerprint(raw / split / 'impressions' / entry['source'])
                    and path.exists() and prior.get('fingerprint') == fingerprint(path)
                    and pq.read_metadata(path).num_rows == entry['candidates'])
        # Expanding a cache is monotonic: a later small run retains verified shards.
        entries = [entry for entry in all_entries if entry['source'] in requested or reusable(entry)]
        saved = {entry['source']:dict(old_parts[entry['source']]) for entry in entries if reusable(entry)}
        pending = [entry for entry in entries if entry['source'] not in saved]
        group_frames = {}
        group_vectors = group_rows = None
        strategy = reader
        if strategy == 'auto':
            strategy = 'sequential' if store.vectors[split].nbytes > 1_500_000_000 else 'mapped'
        for number, entry in enumerate(entries):
            source_path = raw / split / 'impressions' / entry['source']
            stamp = fingerprint(source_path)
            if stamp != entry['source_fingerprint']:
                raise ValueError('raw shard changed since prepared table generation')
            relative = Path(split) / entry['source']
            path = output / relative
            prior = previous.get('splits', {}).get(split, {}).get('parts', {}).get(entry['source'], {})
            reuse = reusable(entry)
            if not reuse:
                part_started = time.monotonic()
                # Reopen the file mapping between small feed blocks. This bounds
                # resident mapped pages on Windows even for the 10 GB train file;
                # copied feature columns do not retain views of the memmap.
                if strategy == 'sequential':
                    if not group_frames:
                        group_entries, pending = pending[:prefetch_parts], pending[prefetch_parts:]
                        group_frames = {item['source']:pl.read_parquet(raw / split / 'impressions' / item['source'],
                                        columns=['feed_id','item_id','item_id__embedding_idx',
                                                 'history__item_id__embedding_idx','user_embedding'])
                                        for item in group_entries}
                        group_rows = np.unique(np.concatenate([required_vector_rows(frame) for frame in group_frames.values()]))
                        group_vectors = prefetch_vectors(raw,split,group_rows)
                    frame = group_frames.pop(entry['source'])
                    features = build_prefetched_part(frame,split,group_vectors,group_rows,batch_size=batch_size)
                    prefetched_bytes = group_vectors.nbytes
                    if not group_frames:
                        group_vectors = group_rows = None
                else:
                    frame = pl.read_parquet(source_path, columns=['feed_id','item_id','item_id__embedding_idx',
                                                                  'history__item_id__embedding_idx','user_embedding'])
                    chunks = []
                    for start in range(0, frame.height, FEED_WINDOW):
                        vector_window = load_embeddings(raw, split)
                        try:
                            chunks.append(build_feature_part(frame.slice(start, FEED_WINDOW), split, vector_window, batch_size=batch_size))
                        finally:
                            vector_window._mmap.close()
                    features = pl.concat(chunks)
                    prefetched_bytes = None
                if features.height != entry['candidates']:
                    raise ValueError('feature count does not match prepared candidates')
                write_parquet_atomic(features, path)
                del frame, features
                production = {'producer_implementation_sha256':implementation_sha,'reader':strategy,
                              'duration_seconds':round(time.monotonic()-part_started,3),'prefetched_bytes':prefetched_bytes}
            else:
                production = {key:prior[key] for key in ('reader','duration_seconds','prefetched_bytes') if key in prior}
                production['producer_implementation_sha256'] = prior.get('producer_implementation_sha256',previous.get('implementation_sha256'))
            saved[entry['source']] = {'path': relative.as_posix(), 'rows': entry['candidates'],
                                      'source': stamp, 'fingerprint': fingerprint(path),**production}
            manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved),
                                         'total_parts': len(source['splits'][split]['parts'])}
            manifest['resources'] = memory_snapshot()
            manifest['duration_seconds'] = round(time.monotonic() - started, 2)
            write_json(output / 'manifest.json', manifest)
            if not reuse or number == 0 or (number + 1) % 16 == 0 or number + 1 == len(entries):
                print(f"cosines {split}: {len(saved)}/{len(entries)} shards; {time.monotonic() - started:.0f}s", flush=True)
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
    if split == 'train' and manifest.get('feature_group') in ('itemstats','userstats'):
        if not set(source_parts) <= set(manifest['excluded_train_sources']):
            raise ValueError(f"{manifest['feature_group']} donor shards overlap requested model training shards")
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
