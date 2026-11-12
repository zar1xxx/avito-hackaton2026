"""Audit userstats cache and independently recompute sampled users from donors."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.itemstats import EVENTS,FIELDS,KEY,training_sources
from src.common.preparation import fingerprint,write_json
from src.common.resources import memory_snapshot
from src.common.userstats import feature_names,validate_features


def empty_stat():
    return {'impressions':0,'clicks':0,'positions':Counter(),'history':Counter()}


def audit(prepared,cache,output,*,samples_per_part=3):
    prepared,cache,output=map(Path,(prepared,cache,output))
    if samples_per_part<1:
        raise ValueError('samples_per_part must be positive')
    manifest_bytes=(cache/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    source_bytes=(prepared/'manifest.json').read_bytes()
    source=json.loads(source_bytes)
    identity=manifest['identity']
    if not source.get('complete') or not manifest.get('complete'):
        raise ValueError('complete prepared data and userstats required')
    if identity['prepared_manifest_sha256']!=hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest changed')
    excluded=set(manifest['excluded_train_sources'])
    donors=set(identity['donor_sources'])
    train={entry['source']:entry for entry in source['splits']['train']['parts']}
    expected_excluded={entry['source'] for entry in training_sources(source,len(excluded))}
    if excluded!=expected_excluded or excluded & donors or excluded|donors!=set(train):
        raise ValueError('invalid donor exclusion contract')
    if manifest['features']!=feature_names() or manifest['key']!=KEY:
        raise ValueError('unexpected userstats feature contract')
    for relative,stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared/relative)!=stamp:
            raise ValueError('userstats source changed')
    report={'complete':False,'scope':'all covered rows: keys, types, finite values, source identities and donor exclusion; independent sampled users from every donor shard',
            'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'excluded_train_parts':len(excluded),
            'donor_parts':len(donors),'samples_per_part':samples_per_part,'sampled_candidate_rows':0,
            'sampled_users':0,'splits':{},'bytes':0,'formula_atol':2e-6,'formula_rtol':2e-6}
    started=time.monotonic()
    samples=[]
    for split in ('train','val','test'):
        entries={entry['source']:entry for entry in source['splits'][split]['parts']}
        parts=manifest['splits'][split]['parts']
        if set(parts)!=(excluded if split=='train' else set(entries)) or manifest['splits'][split]['covered_parts']!=len(parts):
            raise ValueError('incorrect userstats coverage')
        total=0
        for number,(name,part) in enumerate(parts.items(),1):
            entry=entries[name]
            path=cache/part['path']
            if fingerprint(path)!=part['fingerprint'] or fingerprint(prepared/entry['candidate_path'])!=part['source']:
                raise ValueError('userstats output or candidate input changed')
            features=pl.read_parquet(path)
            validate_features(features,entry['candidates'])
            candidates=pl.read_parquet(prepared/entry['candidate_path'],columns=[*KEY,'item_id','user_id']).sort('feed_id','candidate_index')
            if not features.select(*KEY,'item_id').equals(candidates.select(*KEY,'item_id')):
                raise ValueError('userstats candidate occurrence keys changed')
            for index in np.linspace(0,features.height-1,min(samples_per_part,features.height),dtype=int):
                samples.append((candidates['user_id'][int(index)],features.row(int(index),named=True)))
            total+=features.height
            report['bytes']+=path.stat().st_size
            if number==1 or number%16==0 or number==len(parts):
                print(f'userstats audit {split}: {number}/{len(parts)}',flush=True)
        report['splits'][split]={'parts':len(parts),'rows':total}
    known=sorted({user for user,_ in samples})
    stats=defaultdict(empty_stat)
    position_impressions,position_clicks=Counter(),Counter()
    for number,name in enumerate(sorted(donors),1):
        entry=train[name]
        candidates=pl.read_parquet(prepared/entry['candidate_path'],columns=['feed_id','user_id','pos','target'])
        position=candidates.group_by('pos').agg(pl.len().alias('n'),pl.col('target').cast(pl.Int64).sum().alias('clicks'))
        for pos,count,clicks in position.iter_rows():
            position_impressions[pos]+=count
            position_clicks[pos]+=clicks
        selected=candidates.filter(pl.col('user_id').is_in(known)).group_by('user_id','pos').agg(
            pl.len().alias('n'),pl.col('target').cast(pl.Int64).sum().alias('clicks'))
        for user,pos,count,clicks in selected.iter_rows():
            stats[user]['impressions']+=count
            stats[user]['clicks']+=clicks
            stats[user]['positions'][pos]+=count
        feeds=pl.read_parquet(prepared/entry['feed_path'],columns=['feed_id','user_id','history__event_type'])
        mapping=candidates.select('feed_id','user_id').unique()
        expected=feeds.select('feed_id','user_id').cast(mapping.schema)
        if (mapping['feed_id'].is_duplicated().any() or feeds['feed_id'].is_duplicated().any()
                or not mapping.sort('feed_id').equals(expected.sort('feed_id'))):
            raise ValueError('candidate and history users differ within a donor feed')
        selected_feeds=feeds.filter(pl.col('user_id').is_in(known))
        for row in selected_feeds.iter_rows(named=True):
            stats[row['user_id']]['history'].update(row['history__event_type'])
        if number==1 or number%32==0 or number==len(donors):
            print(f'userstats donor audit: {number}/{len(donors)}; {time.monotonic()-started:.0f}s',flush=True)
    impressions=sum(position_impressions.values())
    clicks=sum(position_clicks.values())
    summary=manifest['donor_summary']
    if impressions!=summary['impressions'] or clicks!=summary['clicks']:
        raise ValueError('independent donor totals differ')
    prior=clicks/impressions
    if not np.isclose(prior,summary['prior_ctr'],atol=1e-14,rtol=0):
        raise ValueError('independent prior differs')
    rates={pos:position_clicks[pos]/count for pos,count in position_impressions.items()}
    recorded={row['pos']:row for row in summary['position_ctr']}
    if set(recorded)!=set(rates):
        raise ValueError('independent position coverage differs')
    for pos,rate in rates.items():
        row=recorded[pos]
        if row['impressions']!=position_impressions[pos] or row['clicks']!=position_clicks[pos] or not np.isclose(rate,row['rate'],atol=1e-14,rtol=0):
            raise ValueError('independent position CTR differs')
    smoothing=identity['smoothing']
    for user,actual in samples:
        value=stats[user]
        expected_clicks=sum(count*rates[pos] for pos,count in value['positions'].items())
        denominator=expected_clicks+smoothing*prior
        expected={'impressions':value['impressions'],'clicks':value['clicks'],
                  'ctr':(value['clicks']+smoothing*prior)/(value['impressions']+smoothing),
                  'expected_clicks':expected_clicks,
                  'position_ratio':(value['clicks']+smoothing*prior)/denominator if denominator else 1.,
                  'history_count':sum(value['history'].values()),
                  **{f'history_evt{event}':value['history'][event] for event in EVENTS}}
        for field in FIELDS:
            if not np.isclose(actual[f'is_user_{field}'],expected[field],atol=2e-6,rtol=2e-6):
                raise ValueError(f'independent userstats formula mismatch: {field}')
    if (cache/'manifest.json').read_bytes()!=manifest_bytes:
        raise ValueError('userstats manifest changed during audit')
    for relative,stamp in identity['input_fingerprints'].items():
        if fingerprint(prepared/relative)!=stamp:
            raise ValueError('source changed during independent audit')
    report.update(complete=True,sampled_candidate_rows=len(samples),sampled_users=len(known),
                  donor_impressions=impressions,donor_clicks=clicks,prior_ctr=prior,
                  duration_seconds=round(time.monotonic()-started,2),memory=memory_snapshot(),
                  audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/userstats-008'))
    parser.add_argument('--out',type=Path,default=Path('runs/userstats_008_audit.json'))
    args=parser.parse_args()
    audit(args.prepared,args.cache,args.out)
