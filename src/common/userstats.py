"""User statistics from donor train shards, sharing itemstats formulas."""
import hashlib
import json
import math
from pathlib import Path
import time

import duckdb
import polars as pl
import pyarrow.parquet as pq

from src.common import itemstats
from src.common.itemstats import EVENTS, FIELDS, KEY, lookup_sql, training_sources
from src.common.preparation import fingerprint,write_json,write_parquet_atomic
from src.common.resources import memory_snapshot


def feature_names():
    return [f'is_user_{field}' for field in FIELDS]


def validate_features(frame,expected_rows):
    schema={'split':pl.String,'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
            **{name:pl.Float32 for name in feature_names()}}
    if frame.height!=expected_rows or frame.schema!=schema or frame.select(KEY).is_duplicated().any():
        raise ValueError('invalid userstats schema, count or candidate keys')
    if frame.null_count().to_numpy().any():
        raise ValueError('null userstats value')
    for name in feature_names():
        if not frame[name].is_finite().all() or (frame[name]<0).any():
            raise ValueError(f'invalid userstats value: {name}')
    if (frame['is_user_ctr']>1).any():
        raise ValueError('userstats CTR exceeds one')


def build_features(prepared,output,*,train_parts=8,smoothing=20.,memory_mb=512,threads=1,resume=False):
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    if prepared==output or prepared in output.parents or output in prepared.parents:
        raise ValueError('userstats output must be separate from prepared data')
    if not math.isfinite(smoothing) or smoothing<=0 or memory_mb<128 or threads<1:
        raise ValueError('invalid smoothing or resource parameters')
    source_bytes=(prepared/'manifest.json').read_bytes()
    source=json.loads(source_bytes)
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    selected=training_sources(source,train_parts)
    excluded={entry['source'] for entry in selected}
    donors=[entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    outputs={'train':selected,**{split:source['splits'][split]['parts'] for split in ('val','test')}}
    inputs={}
    for entry in [*donors,*[part for group in outputs.values() for part in group]]:
        for field in ('candidate_path','feed_path'):
            relative=entry[field]
            inputs[relative]=fingerprint(prepared/relative)
    identity={'prepared_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),
              'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'formulas_sha256':hashlib.sha256(Path(itemstats.__file__).read_bytes()).hexdigest(),
              'excluded_train_sources':sorted(excluded),'donor_sources':[entry['source'] for entry in donors],
              'smoothing':smoothing,'input_fingerprints':inputs}
    path=output/'manifest.json'
    previous={}
    if not path.exists() and (output/'_statistics.duckdb').exists():
        raise ValueError('userstats database has no source manifest; choose a new output directory')
    if path.exists():
        if not resume:
            raise FileExistsError('userstats output exists; use --resume')
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous.get('identity')!=identity:
            raise ValueError('userstats sources or configuration changed; choose a new output directory')
    manifest={'format_version':1,'feature_group':'userstats','complete':False,'key':KEY,'features':feature_names(),
              'identity':identity,'excluded_train_sources':sorted(excluded),
              'target_policy':'donor train shards disjoint from model training; val/test targets unused',
              'history_policy':'event occurrences across donor feed histories belonging to the user; repeated events counted',
              'resources_config':{'memory_mb':memory_mb,'threads':threads},
              'splits':previous.get('splits',{}),'tables':previous.get('tables',{})}
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    connection=duckdb.connect(str(output/'_statistics.duckdb'),config={
        'memory_limit':f'{memory_mb}MB','threads':str(threads),'temp_directory':str(output/'_spill'),
        'preserve_insertion_order':'false'})
    try:
        write_json(path,manifest)
        connection.from_parquet([str(prepared/e['candidate_path']) for e in donors]).create_view('donors',replace=True)
        connection.from_parquet([str(prepared/e['feed_path']) for e in donors]).create_view('donor_feeds',replace=True)
        connection.from_parquet([str(prepared/e['candidate_path']) for group in outputs.values() for e in group]).create_view('queries',replace=True)

        def materialize(name,query):
            phase_started=time.monotonic()
            connection.execute(f'CREATE TABLE IF NOT EXISTS {name} AS {query}')
            connection.execute('CHECKPOINT')
            manifest['tables'][name]={'rows':connection.execute(f'SELECT count(*) FROM {name}').fetchone()[0],
                                     'last_call_seconds':round(time.monotonic()-phase_started,3)}
            manifest['stage']=name
            manifest['resources']=memory_snapshot()
            write_json(path,manifest)
            print(f'userstats {name}: {manifest["tables"][name]["rows"]:,} rows; {time.monotonic()-started:.0f}s',flush=True)

        materialize('query_user','SELECT DISTINCT user_id FROM queries')
        invalid=connection.execute('SELECT count(*) FROM donors WHERE user_id IS NULL OR target IS NULL '
                                   'OR target NOT IN (0,1) OR pos IS NULL OR pos<0').fetchone()[0]
        invalid+=connection.execute('SELECT count(*) FROM queries WHERE user_id IS NULL').fetchone()[0]
        invalid+=connection.execute('SELECT count(*) FROM donor_feeds WHERE user_id IS NULL OR history__event_type IS NULL').fetchone()[0]
        if invalid:
            raise ValueError('invalid user identifiers, donor labels, positions or histories')
        materialize('position_ctr','SELECT pos,count(*) impressions,sum(target)::BIGINT clicks,avg(target::DOUBLE) rate FROM donors GROUP BY pos')
        impressions,clicks=connection.execute('SELECT sum(impressions),sum(clicks) FROM position_ctr').fetchone()
        if not impressions:
            raise ValueError('empty donor data')
        prior=float(clicks/impressions)
        manifest['donor_summary']={'parts':len(donors),'impressions':int(impressions),'clicks':int(clicks),'prior_ctr':prior,
            'position_ctr':[dict(zip(('pos','impressions','clicks','rate'),row)) for row in connection.execute('SELECT * FROM position_ctr ORDER BY pos').fetchall()]}
        materialize('user_exposure','SELECT user_id,count(*) impressions,sum(target)::BIGINT clicks,'
                    'sum(p.rate) expected_clicks FROM donors JOIN position_ctr p USING (pos) '
                    'SEMI JOIN query_user USING (user_id) GROUP BY user_id')
        event_counts=', '.join(f'count(*) FILTER (WHERE event_type={event}) history_evt{event}' for event in EVENTS)
        materialize('user_history','SELECT user_id,count(*) history_count,'+event_counts+
                    ' FROM (SELECT user_id,unnest(history__event_type) event_type FROM donor_feeds '
                    'SEMI JOIN query_user USING (user_id)) events GROUP BY user_id')
        materialize('user_lookup',lookup_sql('user',prior,smoothing))
        manifest['statistics_fingerprint']=fingerprint(output/'_statistics.duckdb')
        for split,entries in outputs.items():
            old_parts=previous.get('splits',{}).get(split,{}).get('parts',{})
            def reusable(entry):
                old=old_parts.get(entry['source'],{})
                file=output/split/entry['source']
                return (old.get('source')==fingerprint(prepared/entry['candidate_path']) and file.exists()
                        and old.get('fingerprint')==fingerprint(file) and pq.read_metadata(file).num_rows==entry['candidates'])
            saved={entry['source']:dict(old_parts[entry['source']]) for entry in entries if reusable(entry)}
            for number,entry in enumerate(entries,1):
                part_started=time.monotonic()
                file=output/split/entry['source']
                reuse=reusable(entry)
                if not reuse:
                    candidates=pl.read_parquet(prepared/entry['candidate_path'],columns=[*KEY,'item_id','user_id'])
                    connection.register('batch',candidates.to_arrow())
                    query=('SELECT b.split,b.feed_id,b.candidate_index,b.item_id,'+','.join(feature_names())+
                           ' FROM batch b LEFT JOIN user_lookup USING (user_id) ORDER BY b.feed_id,b.candidate_index')
                    features=pl.from_arrow(connection.execute(query).fetch_arrow_table())
                    connection.unregister('batch')
                    validate_features(features,entry['candidates'])
                    if not features.select(*KEY,'item_id').equals(candidates.sort('feed_id','candidate_index').select(*KEY,'item_id')):
                        raise ValueError('userstats keys differ from prepared candidates')
                    write_parquet_atomic(features,file)
                saved[entry['source']]={'path':file.relative_to(output).as_posix(),'rows':entry['candidates'],
                    'source':fingerprint(prepared/entry['candidate_path']),'fingerprint':fingerprint(file),
                    'duration_seconds':old_parts[entry['source']].get('duration_seconds',0.) if reuse else round(time.monotonic()-part_started,3)}
                manifest['splits'][split]={'parts':saved,'covered_parts':len(saved),'total_parts':len(source['splits'][split]['parts'])}
                manifest['stage']=f'features_{split}'
                manifest['resources']=memory_snapshot()
                write_json(path,manifest)
                if not reuse or number==len(entries):
                    print(f'userstats {split}: {number}/{len(entries)} shards; {time.monotonic()-started:.0f}s',flush=True)
        for relative,stamp in inputs.items():
            if fingerprint(prepared/relative)!=stamp:
                raise ValueError('source changed during userstats generation')
        manifest.update(complete=True,stage='complete',all_train_parts=False,
                        duration_seconds=round(time.monotonic()-started,2),resources=memory_snapshot())
        write_json(path,manifest)
        return manifest
    finally:
        connection.close()
