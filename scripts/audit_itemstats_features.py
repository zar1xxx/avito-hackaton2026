"""Audit every itemstats row and recompute sampled lookups from donor Parquet."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.itemstats import EVENTS, FIELDS, KEY, feature_names, training_sources, validate_features
from src.common.preparation import fingerprint,write_json
from src.common.resources import memory_snapshot


def empty_stat():
    return {'impressions':0,'clicks':0,'positions':Counter(),'history':Counter()}


def audit(prepared,cache,output,*,samples_per_part=3):
    prepared,cache,output=map(Path,(prepared,cache,output))
    if samples_per_part < 1:
        raise ValueError('samples_per_part must be positive')
    manifest_bytes=(cache/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    source_bytes=(prepared/'manifest.json').read_bytes()
    source=json.loads(source_bytes)
    identity=manifest['identity']
    if not manifest.get('complete') or not source.get('complete'):
        raise ValueError('complete itemstats and prepared data required')
    if identity['prepared_manifest_sha256']!=hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest differs from statistics source')
    excluded=set(manifest['excluded_train_sources'])
    donors=set(identity['donor_sources'])
    all_train={entry['source']:entry for entry in source['splits']['train']['parts']}
    expected_excluded={entry['source'] for entry in training_sources(source,len(excluded))}
    if excluded!=expected_excluded or donors & excluded or donors|excluded!=set(all_train):
        raise ValueError('invalid donor exclusion contract')
    if manifest['features']!=feature_names() or manifest['key']!=KEY:
        raise ValueError('unexpected itemstats feature contract')
    for relative,stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared/relative)!=stamp:
            raise ValueError('itemstats source changed')
    report={'complete':False,'scope':'all cache rows: keys, schema, finite values, source identities and donor exclusion; independent sampled lookup values from all donor shards',
            'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'donor_parts':len(donors),
            'excluded_train_parts':len(excluded),'samples_per_part':samples_per_part,
            'sampled_candidate_rows':0,'splits':{},'bytes':0,'formula_rtol':2e-6,'formula_atol':2e-6}
    started=time.monotonic()
    samples=[]
    known={'item':set(),'category':set()}
    for split in ('train','val','test'):
        entries={entry['source']:entry for entry in source['splits'][split]['parts']}
        parts=manifest['splits'][split]['parts']
        expected_names=excluded if split=='train' else set(entries)
        if set(parts)!=expected_names or manifest['splits'][split]['covered_parts']!=len(parts):
            raise ValueError(f'{split}: wrong itemstats coverage')
        total=0
        for number,(name,part) in enumerate(parts.items(),1):
            entry=entries[name]
            path=cache/part['path']
            if fingerprint(path)!=part['fingerprint'] or fingerprint(prepared/entry['candidate_path'])!=part['source']:
                raise ValueError('changed feature or source file')
            features=pl.read_parquet(path)
            validate_features(features,entry['candidates'])
            candidates=pl.read_parquet(prepared/entry['candidate_path'],columns=[*KEY,'item_id','category_id']).sort('feed_id','candidate_index')
            if not features.select(*KEY,'item_id').equals(candidates.select(*KEY,'item_id')):
                raise ValueError('itemstats keys differ from prepared candidates')
            for index in np.linspace(0,features.height-1,min(samples_per_part,features.height),dtype=int):
                row=candidates.row(int(index),named=True)
                values=features.row(int(index),named=True)
                samples.append((row['item_id'],row['category_id'],values))
                known['item'].add(row['item_id'])
                known['category'].add(row['category_id'])
            total+=features.height
            report['bytes']+=path.stat().st_size
            if number==1 or number%16==0 or number==len(parts):
                print(f'itemstats audit {split}: {number}/{len(parts)}',flush=True)
        report['splits'][split]={'parts':len(parts),'rows':total}
    stats={entity:defaultdict(empty_stat) for entity in known}
    position_impressions,position_clicks=Counter(),Counter()
    for number,name in enumerate(sorted(donors),1):
        entry=all_train[name]
        candidates=pl.read_parquet(prepared/entry['candidate_path'],columns=['item_id','category_id','pos','target'])
        counts=candidates.group_by('pos').agg(pl.len().alias('n'),pl.col('target').cast(pl.Int64).sum().alias('clicks'))
        for pos,impressions,clicks in counts.iter_rows():
            position_impressions[pos]+=impressions
            position_clicks[pos]+=clicks
        for entity,identifiers in known.items():
            key=f'{entity}_id'
            counts=candidates.filter(pl.col(key).is_in(sorted(identifiers))).group_by(key,'pos').agg(
                pl.len().alias('n'),pl.col('target').cast(pl.Int64).sum().alias('clicks'))
            for identifier,pos,impressions,clicks in counts.iter_rows():
                value=stats[entity][identifier]
                value['impressions']+=impressions
                value['clicks']+=clicks
                value['positions'][pos]+=impressions
        feeds=pl.read_parquet(prepared/entry['feed_path'],columns=['history__item_id','history__category_id','history__event_type'])
        history=feeds.rename({'history__item_id':'item_id','history__category_id':'category_id','history__event_type':'event_type'})\
            .explode('item_id','category_id','event_type').filter(pl.col('item_id').is_not_null())
        for entity,identifiers in known.items():
            key=f'{entity}_id'
            counts=history.filter(pl.col(key).is_in(sorted(identifiers))).group_by(key,'event_type').len()
            for identifier,event,count in counts.iter_rows():
                stats[entity][identifier]['history'][event]+=count
        if number==1 or number%32==0 or number==len(donors):
            print(f'itemstats donor audit: {number}/{len(donors)}; {time.monotonic()-started:.0f}s',flush=True)
    impressions=sum(position_impressions.values())
    clicks=sum(position_clicks.values())
    summary=manifest['donor_summary']
    if impressions!=summary['impressions'] or clicks!=summary['clicks']:
        raise ValueError('independent donor totals differ')
    prior=clicks/impressions
    rates={pos:position_clicks[pos]/count for pos,count in position_impressions.items()}
    recorded={row['pos']:row for row in summary['position_ctr']}
    if set(recorded)!=set(rates) or not np.isclose(prior,summary['prior_ctr'],atol=1e-14,rtol=0):
        raise ValueError('independent position baseline differs')
    for pos,rate in rates.items():
        row=recorded[pos]
        if (row['impressions']!=position_impressions[pos] or row['clicks']!=position_clicks[pos]
                or not np.isclose(rate,row['rate'],atol=1e-14,rtol=0)):
            raise ValueError('independent position CTR differs')
    smoothing=identity['smoothing']
    for item,category,row in samples:
        for entity,identifier in [('item',item),('category',category)]:
            value=stats[entity][identifier]
            expected_clicks=sum(count*rates[pos] for pos,count in value['positions'].items())
            denominator=expected_clicks+smoothing*prior
            expected={'impressions':value['impressions'],'clicks':value['clicks'],
                'ctr':(value['clicks']+smoothing*prior)/(value['impressions']+smoothing),
                'expected_clicks':expected_clicks,
                'position_ratio':(value['clicks']+smoothing*prior)/denominator if denominator else 1.,
                'history_count':sum(value['history'].values()),
                **{f'history_evt{event}':value['history'][event] for event in EVENTS}}
            for field in FIELDS:
                column=f'is_{entity}_{field}'
                if not np.isclose(row[column],expected[field],atol=2e-6,rtol=2e-6):
                    raise ValueError(f'independent itemstats formula mismatch: {column}')
    if (cache/'manifest.json').read_bytes()!=manifest_bytes:
        raise ValueError('itemstats manifest changed during audit')
    for relative,stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared/relative)!=stamp:
            raise ValueError('source changed during independent audit')
    report.update(complete=True,sampled_candidate_rows=len(samples),donor_impressions=impressions,donor_clicks=clicks,
                  prior_ctr=prior,duration_seconds=round(time.monotonic()-started,2),memory=memory_snapshot(),
                  audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/itemstats-008'))
    parser.add_argument('--out',type=Path,default=Path('runs/itemstats_008_audit.json'))
    args=parser.parse_args()
    audit(args.prepared,args.cache,args.out)
