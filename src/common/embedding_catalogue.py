"""One item catalogue with explicit train/val/test vector versions.

Raw embedding files remain the backing vector storage; dense item->row indices
and a Parquet catalogue provide a shared API without copying the 11 GB vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.data import load_embeddings, list_parts
from src.common.preparation import fingerprint, flat_values, write_json, write_parquet_atomic


SPLITS = ('train', 'val', 'test')


def catalogue_signature(manifest):
    """Stable content identity, independent of audit duration and output mtimes."""
    content = {name: manifest[name] for name in ('format_version', 'max_item_id', 'key')}
    content['sources'] = {split: value['source'] for split, value in manifest['splits'].items()}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode('utf-8')).hexdigest()


def file_signature(root, split, parts):
    return {'parts': {path.name: fingerprint(path) for path in parts},
            'meta': hashlib.sha256((root / split / 'item_embeddings/data.meta').read_bytes()).hexdigest(),
            'vectors': fingerprint(root / split / 'item_embeddings/data.memmap')}


def update_index(index, ids, rows, vectors):
    """Preserve the first valid version and reject conflicting versions inside a split."""
    valid = (ids > 0) & (rows >= 0)
    ids, rows = ids[valid], rows[valid]
    if not len(ids):
        return 0
    if ids.max() >= len(index) or rows.max() >= len(vectors):
        raise ValueError('catalogue ID or vector index out of bounds')
    unique_ids, first = np.unique(ids, return_index=True)
    old = index[unique_ids]
    new = old < 0
    index[unique_ids[new]] = rows[first[new]]
    aliases = index[ids] != rows
    if not aliases.any():
        return 0
    pairs = np.unique(np.column_stack([index[ids[aliases]], rows[aliases]]), axis=0)
    for start in range(0, len(pairs), 16384):
        pair = pairs[start:start + 16384]
        if not np.array_equal(vectors[pair[:, 0]], vectors[pair[:, 1]]):
            raise ValueError('same item has different vectors within one split; a finer version key is required')
    return len(pairs)


def compare_versions(left, right, left_vectors, right_vectors, batch_size=16384):
    """Exact byte equality plus cosine drift on all matched items, in small batches."""
    common = (left >= 0) & (right >= 0)
    a, b = left[common], right[common]
    result = {'compared_items': len(a), 'identical': 0, 'different': 0,
              'cosine_sum': 0., 'cosine_min': 1., 'cosine_max': -1., 'zero_norm_pairs': 0}
    for start in range(0, len(a), batch_size):
        av = left_vectors[a[start:start + batch_size]]
        bv = right_vectors[b[start:start + batch_size]]
        equal = np.all(av == bv, axis=1)
        result['identical'] += int(equal.sum())
        result['different'] += int((~equal).sum())
        af, bf = av.astype(np.float32), bv.astype(np.float32)
        norm = np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1)
        score = np.divide(np.einsum('ij,ij->i', af, bf), norm, out=np.zeros(len(af), dtype=np.float32), where=norm > 0)
        result['zero_norm_pairs'] += int((norm == 0).sum())
        result['cosine_sum'] += float(score.sum(dtype=np.float64))
        result['cosine_min'] = min(result['cosine_min'], float(score.min()))
        result['cosine_max'] = max(result['cosine_max'], float(score.max()))
    return result


def stream_compare(left_indices, right_indices, raw, left_split, right_split):
    """Stream the large left file in row order; only the small right split is resident."""
    left, right = np.concatenate(left_indices), np.concatenate(right_indices)
    order = np.argsort(left)
    left, right = left[order], right[order]
    del order
    result = {'compared_items': len(left), 'identical': 0, 'different': 0,
              'cosine_sum': 0., 'cosine_min': 1., 'cosine_max': -1., 'zero_norm_pairs': 0}
    if not len(left):
        return result
    left_meta = json.loads((raw / left_split / 'item_embeddings/data.meta').read_text(encoding='utf-8'))
    right_meta = json.loads((raw / right_split / 'item_embeddings/data.meta').read_text(encoding='utf-8'))
    right_vectors = np.fromfile(raw / right_split / 'item_embeddings/data.memmap', dtype=np.int8).reshape(right_meta['shape'])
    block_rows = 65536
    buffer = np.empty((block_rows, 128), dtype=np.int8)
    blocks = np.unique(left // block_rows)
    with (raw / left_split / 'item_embeddings/data.memmap').open('rb') as source:
        for number, block in enumerate(blocks):
            first_row = int(block) * block_rows
            start, end = np.searchsorted(left, [first_row, first_row + block_rows])
            source.seek(first_row * 128)
            size = source.readinto(buffer)
            expected = min(block_rows, left_meta['shape'][0] - first_row) * 128
            if size < expected:
                raise ValueError('truncated embedding file during sequential audit')
            av = buffer[left[start:end] - first_row]
            bv = right_vectors[right[start:end]]
            equal = np.all(av == bv, axis=1)
            result['identical'] += int(equal.sum())
            result['different'] += int((~equal).sum())
            af, bf = av.astype(np.float32), bv.astype(np.float32)
            norm = np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1)
            scores = np.divide(np.einsum('ij,ij->i', af, bf), norm,
                               out=np.zeros(len(af), dtype=np.float32), where=norm > 0)
            result['zero_norm_pairs'] += int((norm == 0).sum())
            result['cosine_sum'] += float(scores.sum(dtype=np.float64))
            result['cosine_min'] = min(result['cosine_min'], float(scores.min()))
            result['cosine_max'] = max(result['cosine_max'], float(scores.max()))
            if number == 0 or (number + 1) % 128 == 0 or number + 1 == len(blocks):
                print(f'vector audit {left_split}/{right_split}: {number + 1}/{len(blocks)} sequential blocks', flush=True)
    return result


def build_catalogue(raw: Path, output: Path, *, resume=False):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError('raw and catalogue directories must be separate')
    previous = {}
    if (output / 'manifest.json').exists():
        if not resume:
            raise FileExistsError('catalogue exists; use --resume')
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    parts = {split: list_parts(raw, split) for split in SPLITS}
    maximum = 0
    for split in SPLITS:
        limits = pl.scan_parquet(parts[split]).select(
            pl.col('item_id').list.max().max().alias('candidate'),
            pl.col('history__item_id').list.max().max().alias('history')).collect(streaming=True).row(0)
        maximum = max(maximum, *(int(value) for value in limits if value is not None))
    if maximum >= np.iinfo(np.int32).max:
        raise ValueError('item IDs exceed int32 catalogue range')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'format_version': 1, 'complete': False, 'max_item_id': maximum, 'index_dtype': 'int32',
                'key': ['split', 'item_id'], 'raw_root_relative': os.path.relpath(raw, output),
                'policy': 'preserve per-split vectors; never overwrite an earlier split with a later vector',
                'storage': 'Parquet item catalogue + dense item-to-row indices; original memmaps back all vectors',
                'splits': {}}
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    indices, vectors = {}, {}
    for split in SPLITS:
        signature = file_signature(raw, split, parts[split])
        path = output / f'{split}.index.memmap'
        prior = previous.get('splits', {}).get(split, {})
        reuse = (resume and prior.get('complete') and previous.get('max_item_id') == maximum
                 and prior.get('source') == signature and path.exists()
                 and prior.get('index_fingerprint') == fingerprint(path)
                 and path.stat().st_size == (maximum + 1) * 4)
        index = np.memmap(path, dtype=np.int32, mode='r+' if reuse else 'w+', shape=(maximum + 1,))
        vector = load_embeddings(raw, split)
        aliases = 0
        if not reuse:
            index[:] = -1
            for number, source in enumerate(parts[split]):
                frame = pl.read_parquet(source, columns=['item_id', 'item_id__embedding_idx',
                                                         'history__item_id', 'history__item_id__embedding_idx'])
                for prefix in ('', 'history__'):
                    ids = flat_values(frame[prefix + 'item_id'])
                    rows = flat_values(frame[prefix + 'item_id__embedding_idx'])
                    aliases += update_index(index, ids, rows, vector)
                if number == 0 or (number + 1) % 32 == 0 or number + 1 == len(parts[split]):
                    print(f"catalogue {split}: {number + 1}/{len(parts[split])} shards, {time.monotonic() - started:.0f}s", flush=True)
            index.flush()
        else:
            aliases = prior['identical_vector_index_aliases']
            print(f'catalogue {split}: verified index reused', flush=True)
        indices[split], vectors[split] = index, vector
        manifest['splits'][split] = {'complete': True, 'source': signature, 'index': path.name,
                                     'index_fingerprint': fingerprint(path), 'identical_vector_index_aliases': aliases}
        write_json(output / 'manifest.json', manifest)
    pairs = {f'{left}/{right}': ([], []) for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    valid_counts = {split: 0 for split in SPLITS}
    catalogue_rows = 0
    item_parts = []
    for start in range(0, maximum + 1, 1_000_000):
        end = min(start + 1_000_000, maximum + 1)
        chunk = {split: np.asarray(index[start:end]) for split, index in indices.items()}
        present = np.logical_or.reduce([value >= 0 for value in chunk.values()])
        if start == 0:
            present[0] = False
        if present.any():
            table = pl.DataFrame({'item_id': np.arange(start, end, dtype=np.int32)[present],
                                  **{f'{split}_embedding_idx': value[present] for split, value in chunk.items()}})
            path = output / 'items' / f'part-{start // 1_000_000:05d}.parquet'
            write_parquet_atomic(table, path)
            item_parts.append(path.relative_to(output).as_posix())
            catalogue_rows += table.height
        for split, value in chunk.items():
            valid_counts[split] += int((value >= 0).sum())
        for pair, lists in pairs.items():
            left, right = pair.split('/')
            common = (chunk[left] >= 0) & (chunk[right] >= 0)
            lists[0].append(chunk[left][common].copy())
            lists[1].append(chunk[right][common].copy())
        if start % 16_000_000 == 0:
            print(f'catalogue audit: IDs {end:,}/{maximum + 1:,}', flush=True)
    # Release the ~1.3 GB dense-index mappings before reading vector files.
    del chunk, value
    for mapped in [*indices.values(), *vectors.values()]:
        mapped._mmap.close()
    indices.clear()
    vectors.clear()
    totals = {pair: stream_compare(left, right, raw, *pair.split('/')) for pair, (left, right) in pairs.items()}
    for total in totals.values():
        count = total['compared_items']
        total['different_fraction'] = total['different'] / count if count else 0.
        total['cosine_mean'] = total.pop('cosine_sum') / count if count else None
        if not count:
            total['cosine_min'] = total['cosine_max'] = None
    manifest.update(complete=True, catalogue_items=catalogue_rows, valid_items_by_split=valid_counts,
                    item_parts=item_parts, audit=totals, duration_seconds=round(time.monotonic() - started, 2))
    write_json(output / 'manifest.json', manifest)
    print(f'Catalogue complete: {catalogue_rows:,} items; {output}', flush=True)
    return manifest


class EmbeddingCatalogue:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if not self.manifest.get('complete'):
            raise ValueError('embedding catalogue is incomplete')
        self.raw = (self.root / self.manifest['raw_root_relative']).resolve()
        self.indices = {split: np.memmap(self.root / info['index'], dtype=np.int32, mode='r',
                                        shape=(self.manifest['max_item_id'] + 1,))
                        for split, info in self.manifest['splits'].items()}
        self.vectors = {split: load_embeddings(self.raw, split) for split in SPLITS}

    def lookup(self, split, item_ids):
        ids = np.asarray(item_ids, dtype=np.int64)
        output = np.full(ids.shape, -1, dtype=np.int32)
        valid = (ids > 0) & (ids <= self.manifest['max_item_id'])
        output[valid] = self.indices[split][ids[valid]]
        return output

    def gather(self, split, item_ids, original_missing=None):
        """Lookup the requested split version; optionally retain original -1 masks."""
        indices = self.lookup(split, item_ids)
        valid = indices >= 0
        if original_missing is not None:
            valid &= ~np.asarray(original_missing, dtype=bool)
        output = np.zeros(indices.shape + (128,), dtype=np.float32)
        output[valid] = self.vectors[split][indices[valid]].astype(np.float32)
        return output
