"""Группа признаков «history»: связь кандидата с историей пользователя (задачи Антона).

Используются только поля истории самой ленты (события до показа) и item_id/category_id кандидата.
Target, pos и другие ленты не используются — утечки таргета нет.
История идёт от новых событий к старым, индекс 0 = самое свежее.

Формат кеша тот же, что у групп Димы (cosine, infeed): manifest.json + <split>/<shard>.parquet
с ключом (split, feed_id, candidate_index) и item_id, поэтому подключается через
src.common.similarity.attach_features и флаг раннера --feature-root data/features/history.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.preparation import fingerprint, write_json, write_parquet_atomic

KEY = ['split', 'feed_id', 'candidate_index']
EVENT_TYPES = list(range(1, 11))   # по EDA в истории встречаются типы 1..10
RECENT = (10, 50)
MAX_HISTORY = 256


def feature_names():
    names = ['h_item_cnt', 'h_item_last', 'h_item_last_evt', 'h_item_evt_nunique']
    names += [f'h_item_evt{e}' for e in EVENT_TYPES]
    names += ['h_cat_share', 'h_cat_last', 'h_cat_is_last', 'h_cat_rank', 'h_cat_wshare']
    names += [f'h_cat_share_r{n}' for n in RECENT]
    names += [f'h_cat_evt{e}' for e in EVENT_TYPES]
    names += ['h_ncats', 'h_top_cat_share', 'h_nitems_share']
    names += [f'h_user_evt{e}' for e in EVENT_TYPES]
    names += ['h_item_cnt_feedrank', 'h_cat_share_feedrank']
    return names


def _feed_block(cand_item, cand_cat, h_item, h_cat, h_evt):
    """Признаки для кандидатов одной ленты -> dict имя -> массив длины M (float32)."""
    m, h = len(cand_item), len(h_item)
    out = {name: np.zeros(m, dtype=np.float32) for name in feature_names()}
    out['h_item_last'][:] = 1.0      # 1 = товара в истории нет; иначе позиция/256 (0 = самое свежее)
    out['h_cat_last'][:] = 1.0
    if h == 0:
        return out
    inv_h = 1.0 / h
    evt_onehot = (h_evt[:, None] == np.asarray(EVENT_TYPES)[None, :]).astype(np.float32)   # (H, E)
    decay = np.exp(-np.arange(h, dtype=np.float32) / 30.0)

    eq = cand_item[:, None] == h_item[None, :]                                             # (M, H)
    cnt = eq.sum(1)
    seen = cnt > 0
    first = eq.argmax(1)
    out['h_item_cnt'] = cnt.astype(np.float32)
    out['h_item_last'] = np.where(seen, first / MAX_HISTORY, 1.0).astype(np.float32)
    out['h_item_last_evt'] = np.where(seen, h_evt[first], 0).astype(np.float32)
    item_evt = eq.astype(np.float32) @ evt_onehot                                          # (M, E)
    out['h_item_evt_nunique'] = (item_evt > 0).sum(1).astype(np.float32)
    for j, e in enumerate(EVENT_TYPES):
        out[f'h_item_evt{e}'] = item_evt[:, j]

    ceq = cand_cat[:, None] == h_cat[None, :]
    ccnt = ceq.sum(1)
    cseen = ccnt > 0
    out['h_cat_share'] = (ccnt * inv_h).astype(np.float32)
    out['h_cat_last'] = np.where(cseen, ceq.argmax(1) / MAX_HISTORY, 1.0).astype(np.float32)
    out['h_cat_is_last'] = (cand_cat == h_cat[0]).astype(np.float32)
    out['h_cat_wshare'] = (ceq @ decay / decay.sum()).astype(np.float32)
    for n in RECENT:
        k = min(n, h)
        out[f'h_cat_share_r{n}'] = (ceq[:, :k].sum(1) / k).astype(np.float32)
    cat_evt = ceq.astype(np.float32) @ evt_onehot
    evt_tot = np.maximum(evt_onehot.sum(0), 1.0)
    for j, e in enumerate(EVENT_TYPES):
        out[f'h_cat_evt{e}'] = cat_evt[:, j] / evt_tot[j]      # доля событий типа e, пришедшихся на категорию

    cats, cat_counts = np.unique(h_cat, return_counts=True)
    order = np.argsort(-cat_counts, kind='stable')
    rank_of = {int(c): r for r, c in enumerate(cats[order])}
    out['h_cat_rank'] = np.array([rank_of.get(int(c), len(cats)) for c in cand_cat], dtype=np.float32)
    out['h_ncats'][:] = len(cats)
    out['h_top_cat_share'][:] = cat_counts.max() * inv_h
    out['h_nitems_share'][:] = len(np.unique(h_item)) * inv_h
    user_evt = evt_onehot.mean(0)
    for j, e in enumerate(EVENT_TYPES):
        out[f'h_user_evt{e}'][:] = user_evt[j]

    for src, dst in (('h_item_cnt', 'h_item_cnt_feedrank'), ('h_cat_share', 'h_cat_share_feedrank')):
        v = out[src]
        out[dst] = ((v[:, None] > v[None, :]).sum(1) + 0.5 * (v[:, None] == v[None, :]).sum(1)).astype(np.float32) / m
    return out


def build_feature_part(candidates: pl.DataFrame, feeds: pl.DataFrame) -> pl.DataFrame:
    """candidates: split, feed_id, candidate_index, item_id, category_id; feeds: feed_id + history__*."""
    candidates = candidates.select(*KEY, 'item_id', 'category_id').sort('feed_id', 'candidate_index')
    if candidates.select(KEY).is_duplicated().any():
        raise ValueError('duplicate candidate occurrence key')
    hist = {int(r[0]): r[1:] for r in feeds.select(
        'feed_id', 'history__item_id', 'history__category_id', 'history__event_type').iter_rows()}
    feed_ids = candidates['feed_id'].to_numpy()
    items = candidates['item_id'].to_numpy().astype(np.int64)
    cats = candidates['category_id'].to_numpy().astype(np.int64)
    bounds = np.flatnonzero(np.diff(feed_ids)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(feed_ids)]])
    names = feature_names()
    cols = {name: np.empty(len(feed_ids), dtype=np.float32) for name in names}
    for a, b in zip(starts, ends):
        fid = int(feed_ids[a])
        if fid not in hist:
            raise ValueError(f'feed {fid} has no row in the feeds table')
        h_item, h_cat, h_evt = (np.asarray(x if x is not None else [], dtype=np.int64) for x in hist[fid])
        if not (len(h_item) == len(h_cat) == len(h_evt)):
            raise ValueError(f'feed {fid}: history arrays differ in length')
        block = _feed_block(items[a:b], cats[a:b], h_item, h_cat, h_evt)
        for name in names:
            cols[name][a:b] = block[name]
    result = candidates.select(*KEY, 'item_id').with_columns([pl.Series(n, cols[n]) for n in names])
    for name in names:
        if not result[name].is_finite().all():
            raise ValueError(f'invalid history feature: {name}')
    return result


def build_features(prepared, output, *, train_parts=None, resume=False):
    """Тот же протокол, что у infeed: манифест, отпечатки файлов, возобновление по шардам."""
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('feature output and prepared input must be separate directories')
    previous = {}
    if (output / 'manifest.json').exists():
        if not resume:
            raise FileExistsError('history output exists; use --resume')
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source_path = prepared / 'manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {'format_version': 1, 'feature_group': 'history', 'complete': False, 'key': KEY,
                'features': feature_names(), 'implementation_sha256': implementation,
                'source_manifest_sha256': source_sha,
                'target_policy': 'only the history arrays of the same feed and candidate item/category; '
                                 'no target, original pos or other feeds',
                'splits': {}}
    write_json(output / 'manifest.json', manifest)
    started = time.monotonic()
    same_code = (resume and previous.get('implementation_sha256') == implementation
                 and previous.get('source_manifest_sha256') == source_sha)
    for split in ('train', 'val', 'test'):
        all_entries = source['splits'][split]['parts']
        entries = all_entries
        if split == 'train' and train_parts is not None:
            if not 1 <= train_parts <= len(entries):
                raise ValueError('train_parts outside source range')
            entries = [entries[int(i)] for i in np.linspace(0, len(entries) - 1, train_parts, dtype=int)]
        prior_parts = previous.get('splits', {}).get(split, {}).get('parts', {}) if same_code else {}
        saved = dict(prior_parts) if split == 'train' else {}   # расширение кеша не теряет готовые шарды
        for number, entry in enumerate(entries):
            cand_path = prepared / entry['candidate_path']
            stamp = fingerprint(cand_path)
            path = output / split / entry['source']
            prior = prior_parts.get(entry['source'], {})
            reuse = (prior.get('source') == stamp and path.exists() and prior.get('fingerprint') == fingerprint(path)
                     and pq.read_metadata(path).num_rows == entry['candidates'])
            if not reuse:
                features = build_feature_part(
                    pl.read_parquet(cand_path, columns=[*KEY, 'item_id', 'category_id']),
                    pl.read_parquet(prepared / entry['feed_path'], columns=[
                        'feed_id', 'history__item_id', 'history__category_id', 'history__event_type']))
                if features.height != entry['candidates']:
                    raise ValueError('feature rows differ from prepared candidates')
                write_parquet_atomic(features, path)
                del features
            saved[entry['source']] = {'path': path.relative_to(output).as_posix(), 'rows': entry['candidates'],
                                      'source': stamp, 'fingerprint': fingerprint(path)}
            if number == 0 or (number + 1) % 8 == 0 or number + 1 == len(entries):
                print(f'history {split}: {number + 1}/{len(entries)} shards; {time.monotonic() - started:.0f}s',
                      flush=True)
                manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved),
                                             'total_parts': len(all_entries)}
                write_json(output / 'manifest.json', manifest)
        manifest['splits'][split] = {'parts': saved, 'covered_parts': len(saved), 'total_parts': len(all_entries)}
    manifest['complete'] = True
    manifest['all_train_parts'] = manifest['splits']['train']['covered_parts'] == manifest['splits']['train']['total_parts']
    manifest['duration_seconds'] = round(time.monotonic() - started, 2)
    write_json(output / 'manifest.json', manifest)
    print(f'History features complete: {output}', flush=True)
    return manifest
