"""Audit history cache files and independently recompute sampled feed features."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.history import EVENT_TYPES, KEY, feature_names
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot


def reference_features(candidates, history):
    items, categories, events = history
    length = len(items)
    category_counts = Counter(categories)
    ordered_categories = sorted(category_counts, key=lambda category: (-category_counts[category],category))
    weights = [math.exp(-index/30) for index in range(length)]
    rows = []
    for item, category in candidates:
        values = dict.fromkeys(feature_names(),0.)
        values.update(h_item_last=1.,h_cat_last=1.)
        if length:
            matches = [index for index,value in enumerate(items) if value == item]
            category_matches = [index for index,value in enumerate(categories) if value == category]
            values.update(h_item_cnt=len(matches),h_item_last=matches[0]/256 if matches else 1.,
                          h_item_last_evt=events[matches[0]] if matches else 0.,
                          h_item_evt_nunique=len({events[index] for index in matches if events[index] in EVENT_TYPES}),
                          h_cat_share=len(category_matches)/length,
                          h_cat_last=category_matches[0]/256 if category_matches else 1.,
                          h_cat_is_last=float(category == categories[0]),
                          h_cat_rank=ordered_categories.index(category) if category in category_counts else len(category_counts),
                          h_cat_wshare=sum(weights[index] for index in category_matches)/sum(weights),
                          h_ncats=len(category_counts),h_top_cat_share=max(category_counts.values())/length,
                          h_nitems_share=len(set(items))/length)
            for window in (10,50):
                values[f'h_cat_share_r{window}'] = categories[:window].count(category)/min(window,length)
            for event in EVENT_TYPES:
                total = events.count(event)
                values[f'h_item_evt{event}'] = sum(events[index] == event for index in matches)
                values[f'h_cat_evt{event}'] = sum(events[index] == event for index in category_matches)/max(total,1)
                values[f'h_user_evt{event}'] = total/length
        rows.append(values)
    if length:
        for source,destination in [('h_item_cnt','h_item_cnt_feedrank'),('h_cat_share','h_cat_share_feedrank')]:
            for row in rows:
                row[destination] = (sum(other[source] < row[source] for other in rows)
                                    + .5*sum(other[source] == row[source] for other in rows))/len(rows)
    return {name:np.asarray([row[name] for row in rows]) for name in feature_names()}


def audit(prepared, cache, output, *, sample_feeds=3):
    prepared,cache,output = map(Path,(prepared,cache,output))
    if sample_feeds < 1:
        raise ValueError('sample_feeds must be positive')
    path = cache/'manifest.json'
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source_bytes = (prepared/'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not manifest.get('complete') or not source.get('complete'):
        raise ValueError('complete input and history cache required')
    if manifest.get('source_manifest_sha256') != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest changed')
    names = feature_names()
    schema = {'split':pl.String,'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
              **{name:pl.Float32 for name in names}}
    if manifest.get('features') != names or manifest.get('key') != KEY:
        raise ValueError('unexpected history feature contract')
    report = {'complete':False,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
              'scope':'all covered rows: keys, schema, finite values and both source fingerprints; independent formulas sampled per shard',
              'sample_feeds_per_shard':sample_feeds,'sampled_feeds':0,'sampled_candidate_rows':0,
              'sampled_repeated_item_feeds':0,'splits':{},'bytes':0,'formula_atol':2e-6,'formula_rtol':2e-6}
    started = time.monotonic()
    for split in ('train','val','test'):
        source_parts = {entry['source']:entry for entry in source['splits'][split]['parts']}
        group = manifest['splits'][split]
        if (group['covered_parts'] != len(group['parts']) or not group['parts']
                or not set(group['parts']) <= set(source_parts)
                or (split != 'train' and set(group['parts']) != set(source_parts))):
            raise ValueError(f'{split}: invalid coverage')
        total = 0
        for number,(name,part) in enumerate(group['parts'].items(),1):
            entry = source_parts[name]
            candidate_path,feed_path = prepared/entry['candidate_path'],prepared/entry['feed_path']
            feature_path = cache/part['path']
            if (fingerprint(candidate_path) != part['source'] or fingerprint(feed_path) != part['feeds_source']
                    or fingerprint(feature_path) != part['fingerprint']):
                raise ValueError(f'{split}: changed source or feature file')
            features = pl.read_parquet(feature_path)
            candidates = pl.read_parquet(candidate_path,columns=[*KEY,'item_id','category_id']).sort('feed_id','candidate_index')
            if (features.schema != schema or features.height != entry['candidates'] or features.height != part['rows']
                    or not features.select(*KEY,'item_id').equals(candidates.select(*KEY,'item_id'))
                    or features.select(KEY).is_duplicated().any()):
                raise ValueError(f'{split}: feature schema, count or candidate keys differ')
            if features.null_count().to_numpy().any():
                raise ValueError(f'{split}: null feature')
            for column in names:
                if not features[column].is_finite().all():
                    raise ValueError(f'{split}: nonfinite feature {column}')
            feeds = pl.read_parquet(feed_path,columns=['feed_id','history__item_id','history__category_id','history__event_type'])
            if feeds['feed_id'].is_duplicated().any():
                raise ValueError('duplicate feed metadata')
            ids = candidates['feed_id'].unique().sort().to_list()
            selected = {ids[index] for index in np.linspace(0,len(ids)-1,min(sample_feeds,len(ids)),dtype=int)}
            repeats = candidates.filter(pl.struct('feed_id','item_id').is_duplicated())['feed_id'].unique().to_list()
            selected.update(repeats)
            histories = {row[0]:row[1:] for row in feeds.iter_rows() if row[0] in selected}
            for feed in sorted(selected):
                history = histories[feed]
                if len({len(values) for values in history}) != 1:
                    raise ValueError('unaligned history arrays')
                rows = candidates.filter(pl.col('feed_id') == feed)
                actual = features.filter(pl.col('feed_id') == feed)
                expected = reference_features(list(rows.select('item_id','category_id').iter_rows()),history)
                for column in names:
                    if not np.allclose(actual[column].to_numpy(),expected[column],atol=2e-6,rtol=2e-6):
                        raise ValueError(f'independent history formula mismatch: {column}')
                report['sampled_feeds'] += 1
                report['sampled_candidate_rows'] += rows.height
                report['sampled_repeated_item_feeds'] += int(feed in repeats)
            total += features.height
            report['bytes'] += feature_path.stat().st_size
            if number == 1 or number % 16 == 0 or number == len(group['parts']):
                print(f'history audit {split}: {number}/{len(group["parts"])}',flush=True)
        report['splits'][split] = {'parts':len(group['parts']),'rows':total}
    if path.read_bytes() != manifest_bytes:
        raise ValueError('history manifest changed during audit')
    report.update(complete=True,duration_seconds=round(time.monotonic()-started,2),memory=memory_snapshot(),
                  audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/history'))
    parser.add_argument('--out',type=Path,default=Path('runs/history_cache_audit.json'))
    args = parser.parse_args()
    audit(args.prepared,args.cache,args.out)
