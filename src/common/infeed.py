"""Target-free statistics calculated only among candidates of the same feed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.preparation import fingerprint, write_json, write_parquet_atomic


FLOAT_COLUMNS = ['f1', 'f2', 'f3', 'f4', 'f5']
KEY = ['split', 'feed_id', 'candidate_index']


def feature_names(columns=FLOAT_COLUMNS):
    return ['feed_size', 'cat_share_feed', *[f'{name}_{suffix}' for name in columns for suffix in ('rank', 'z')]]


def add_infeed_stats(frame, columns=FLOAT_COLUMNS):
    """Ascending average rank / M; population z-score, zero for constant groups.

    Grouping includes split when present. Candidate order, item IDs and labels
    never break ties or influence the feature values.
    """
    groups = ['split', 'feed_id'] if 'split' in frame.columns else ['feed_id']
    size = pl.len().over(groups)
    expressions = [size.cast(pl.Float32).alias('feed_size'),
                   (pl.len().over([*groups, 'category_id']) / size).cast(pl.Float32).alias('cat_share_feed')]
    for name in columns:
        value = pl.col(name)
        mean = value.mean().over(groups)
        std = value.std(ddof=0).over(groups)
        expressions.extend([
            (value.rank('average').over(groups) / size).cast(pl.Float32).alias(f'{name}_rank'),
            pl.when(std > 0).then((value - mean) / std).otherwise(0.).cast(pl.Float32).alias(f'{name}_z'),
        ])
    return frame.with_columns(expressions)


def build_feature_part(frame):
    required = [*KEY, 'item_id', 'category_id', *FLOAT_COLUMNS]
    frame = frame.select(required)
    if not frame.height or frame.null_count().to_numpy().any():
        raise ValueError('empty input or null candidate values')
    if frame.select(KEY).is_duplicated().any():
        raise ValueError('duplicate candidate occurrence key')
    for name in FLOAT_COLUMNS:
        if not frame[name].is_finite().all():
            raise ValueError(f'nonfinite input feature: {name}')
    result = add_infeed_stats(frame).select(*KEY, 'item_id', *feature_names())
    for name in feature_names():
        if result[name].null_count() or not result[name].is_finite().all():
            raise ValueError(f'invalid within-feed feature: {name}')
    return result


def build_features(prepared, output, *, train_parts=None, resume=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('feature output and prepared input must be separate directories')
    previous = {}
    if (output / 'manifest.json').exists():
        if not resume:
            raise FileExistsError('infeed output exists; use --resume')
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source_path = prepared / 'manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {'format_version': 1, 'feature_group': 'infeed', 'complete': False, 'key': KEY,
                'features': feature_names(), 'implementation_sha256': implementation,
                'source_manifest_sha256': source_sha,
                'normalization': {'rank': 'ascending average rank divided by feed size',
                                  'z': 'population std (ddof=0), constant/singleton feeds -> 0'},
                'target_policy': 'only f1-f5 and category_id of the current feed; no target, history or original pos',
                'splits': {}}
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    columns = [*KEY, 'item_id', 'category_id', *FLOAT_COLUMNS]
    for split in ('train', 'val', 'test'):
        all_entries = source['splits'][split]['parts']
        entries = all_entries
        if split == 'train' and train_parts is not None:
            if not 1 <= train_parts <= len(entries):
                raise ValueError('train_parts outside source range')
            entries = [entries[int(i)] for i in np.linspace(0, len(entries) - 1, train_parts, dtype=int)]
        saved = {}
        for number, entry in enumerate(entries):
            input_path = prepared / entry['candidate_path']
            stamp = fingerprint(input_path)
            path = output / split / entry['source']
            prior = previous.get('splits', {}).get(split, {}).get('parts', {}).get(entry['source'], {})
            reuse = (resume and previous.get('implementation_sha256') == implementation
                     and previous.get('source_manifest_sha256') == source_sha and prior.get('source') == stamp
                     and path.exists() and prior.get('fingerprint') == fingerprint(path)
                     and pq.read_metadata(path).num_rows == entry['candidates'])
            if not reuse:
                features = build_feature_part(pl.read_parquet(input_path, columns=columns))
                if features.height != entry['candidates']:
                    raise ValueError('feature rows differ from prepared candidates')
                write_parquet_atomic(features, path)
                del features
            saved[entry['source']] = {'path': path.relative_to(output).as_posix(), 'rows': entry['candidates'],
                                      'source': stamp, 'fingerprint': fingerprint(path)}
            if number == 0 or (number + 1) % 32 == 0 or number + 1 == len(entries):
                print(f'infeed {split}: {number + 1}/{len(entries)} shards; {time.monotonic() - started:.0f}s', flush=True)
                manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved), 'total_parts': len(all_entries)}
                write_json(output / 'manifest.json', manifest)
        manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved), 'total_parts': len(all_entries)}
        write_json(output / 'manifest.json', manifest)
    manifest['complete'] = True
    manifest['all_train_parts'] = manifest['splits']['train']['covered_parts'] == manifest['splits']['train']['total_parts']
    manifest['duration_seconds'] = round(time.monotonic() - started, 2)
    write_json(output / 'manifest.json', manifest)
    print(f'Within-feed features complete: {output}', flush=True)
    return manifest
