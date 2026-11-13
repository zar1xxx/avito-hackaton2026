"""Item/category statistics from train shards excluded from model training."""
import hashlib
import json
import math
from pathlib import Path
import time

import duckdb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.preparation import fingerprint, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot

KEY = ['split', 'feed_id', 'candidate_index']
EVENTS = tuple(range(1, 11))
FIELDS = ['impressions', 'clicks', 'ctr', 'expected_clicks', 'position_ratio',
          'history_count', *[f'history_evt{event}' for event in EVENTS]]


def feature_names():
    return [f'is_{entity}_{field}' for entity in ('item', 'category') for field in FIELDS]


def training_sources(source, train_parts):
    parts = source['splits']['train']['parts']
    if not 1 <= train_parts < len(parts):
        raise ValueError('itemstats requires at least one excluded donor train shard')
    return [parts[int(index)] for index in np.linspace(0, len(parts)-1, train_parts, dtype=int)]


def lookup_sql(entity, prior, smoothing):
    key = f'{entity}_id'
    expressions = {
        'impressions':'coalesce(e.impressions,0)', 'clicks':'coalesce(e.clicks,0)',
        'ctr':f'(coalesce(e.clicks,0)+{smoothing}*{prior})/(coalesce(e.impressions,0)+{smoothing})',
        'expected_clicks':'coalesce(e.expected_clicks,0)',
        'position_ratio':f'coalesce((coalesce(e.clicks,0)+{smoothing}*{prior})/nullif(coalesce(e.expected_clicks,0)+{smoothing}*{prior},0),1)',
        'history_count':'coalesce(h.history_count,0)',
        **{f'history_evt{event}':f'coalesce(h.history_evt{event},0)' for event in EVENTS},
    }
    columns = ', '.join(f'CAST({value} AS FLOAT) AS is_{entity}_{name}' for name,value in expressions.items())
    return (f'SELECT q.{key}, {columns} FROM query_{entity} q '
            f'LEFT JOIN {entity}_exposure e USING ({key}) LEFT JOIN {entity}_history h USING ({key})')


def build_features(prepared, output, *, train_parts=8, smoothing=20., memory_mb=1024, threads=2, resume=False):
    prepared,output = Path(prepared).resolve(),Path(output).resolve()
    if prepared == output or prepared in output.parents or output in prepared.parents:
        raise ValueError('itemstats output must be separate from prepared data')
    if not math.isfinite(smoothing) or smoothing <= 0 or memory_mb < 128 or threads < 1:
        raise ValueError('invalid smoothing or resource parameters')
    source_bytes = (prepared/'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    if not source.get('complete'):
        raise ValueError('prepared dataset incomplete')
    selected = training_sources(source,train_parts)
    excluded = {entry['source'] for entry in selected}
    donors = [entry for entry in source['splits']['train']['parts'] if entry['source'] not in excluded]
    outputs = {'train':selected, **{split:source['splits'][split]['parts'] for split in ('val','test')}}
    inputs = {}
    for entry in [*donors, *[part for group in outputs.values() for part in group]]:
        for field in ('candidate_path','feed_path'):
            relative = entry[field]
            inputs[relative] = fingerprint(prepared/relative)
    identity = {'prepared_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'excluded_train_sources':sorted(excluded),'donor_sources':[entry['source'] for entry in donors],
                'smoothing':smoothing,'input_fingerprints':inputs}
    previous = {}
    manifest_path = output/'manifest.json'
    if not manifest_path.exists() and (output/'_statistics.duckdb').exists():
        raise ValueError('statistics database has no source manifest; choose a new output directory')
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('itemstats output exists; use --resume')
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('identity') != identity:
            raise ValueError('itemstats sources or configuration changed; choose a new output directory')
    manifest = {'format_version':1,'feature_group':'itemstats','complete':False,'key':KEY,
                'features':feature_names(),'identity':identity,'excluded_train_sources':sorted(excluded),
                'target_policy':'train donor shards disjoint from model training; val/test targets unused',
                'history_policy':'event occurrences in donor histories, including repeated events',
                'resources_config':{'memory_mb':memory_mb,'threads':threads},
                'splits':previous.get('splits',{}),'tables':previous.get('tables',{})}
    started = time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    connection = duckdb.connect(str(output/'_statistics.duckdb'),config={
        'memory_limit':f'{memory_mb}MB','threads':str(threads),
        'temp_directory':str(output/'_spill'),'preserve_insertion_order':'false'})
    try:
        write_json(manifest_path,manifest)
        connection.from_parquet([str(prepared/entry['candidate_path']) for entry in donors]).create_view('donors',replace=True)
        connection.from_parquet([str(prepared/entry['feed_path']) for entry in donors]).create_view('donor_feeds',replace=True)
        connection.from_parquet([str(prepared/entry['candidate_path']) for group in outputs.values() for entry in group]).create_view('queries',replace=True)
        connection.execute('CREATE OR REPLACE VIEW history_events AS SELECT unnest(history__item_id) item_id, '
                           'unnest(history__category_id) category_id, unnest(history__event_type) event_type FROM donor_feeds')

        def materialize(name, query):
            phase_started = time.monotonic()
            connection.execute(f'CREATE TABLE IF NOT EXISTS {name} AS {query}')
            connection.execute('CHECKPOINT')
            manifest['tables'][name] = {'rows':connection.execute(f'SELECT count(*) FROM {name}').fetchone()[0],
                                        'last_call_seconds':round(time.monotonic()-phase_started,3)}
            manifest['resources'] = memory_snapshot()
            manifest['stage'] = name
            write_json(manifest_path,manifest)
            print(f'itemstats {name}: {manifest["tables"][name]["rows"]:,} rows; {time.monotonic()-started:.0f}s',flush=True)

        materialize('query_item','SELECT DISTINCT item_id FROM queries')
        materialize('query_category','SELECT DISTINCT category_id FROM queries')
        invalid = connection.execute('SELECT count(*) FROM donors WHERE target IS NULL OR target NOT IN (0,1) OR pos IS NULL OR pos < 0').fetchone()[0]
        if invalid:
            raise ValueError('invalid donor labels or positions')
        invalid = connection.execute('SELECT count(*) FROM donor_feeds WHERE '
                                     'len(history__item_id) != len(history__category_id) OR '
                                     'len(history__item_id) != len(history__event_type)').fetchone()[0]
        if invalid:
            raise ValueError('unaligned donor history arrays')
        materialize('position_ctr','SELECT pos, count(*) impressions, sum(target)::BIGINT clicks, avg(target::DOUBLE) rate FROM donors GROUP BY pos')
        impressions,clicks = connection.execute('SELECT sum(impressions),sum(clicks) FROM position_ctr').fetchone()
        if not impressions:
            raise ValueError('empty donor candidate data')
        prior = float(clicks/impressions)
        manifest['donor_summary'] = {'parts':len(donors),'impressions':int(impressions),'clicks':int(clicks),'prior_ctr':prior,
            'position_ctr':[dict(zip(('pos','impressions','clicks','rate'),row)) for row in connection.execute('SELECT * FROM position_ctr ORDER BY pos').fetchall()]}
        event_counts = ', '.join(f'count(*) FILTER (WHERE event_type={event}) AS history_evt{event}' for event in EVENTS)
        for entity in ('item','category'):
            key = f'{entity}_id'
            subset = f'SEMI JOIN query_item USING ({key})' if entity == 'item' else ''
            materialize(f'{entity}_exposure',f'SELECT {key},count(*) impressions,sum(target)::BIGINT clicks,'
                        f'sum(p.rate) expected_clicks FROM donors JOIN position_ctr p USING (pos) {subset} GROUP BY {key}')
            materialize(f'{entity}_history',f'SELECT {key},count(*) history_count,{event_counts} '
                        f'FROM history_events {subset} GROUP BY {key}')
            materialize(f'{entity}_lookup',lookup_sql(entity,prior,smoothing))
        manifest['statistics_fingerprint'] = fingerprint(output/'_statistics.duckdb')
        for split,entries in outputs.items():
            previous_parts = previous.get('splits',{}).get(split,{}).get('parts',{})
            def reusable(entry):
                old = previous_parts.get(entry['source'],{})
                path = output/split/entry['source']
                return (old.get('source') == fingerprint(prepared/entry['candidate_path'])
                        and path.exists() and old.get('fingerprint') == fingerprint(path)
                        and pq.read_metadata(path).num_rows == entry['candidates'])
            saved = {entry['source']:dict(previous_parts[entry['source']]) for entry in entries if reusable(entry)}
            for number,entry in enumerate(entries,1):
                part_started = time.monotonic()
                path = output/split/entry['source']
                stamp = fingerprint(prepared/entry['candidate_path'])
                old = previous_parts.get(entry['source'],{})
                reuse = reusable(entry)
                if not reuse:
                    candidates = pl.read_parquet(prepared/entry['candidate_path'],columns=[*KEY,'item_id','category_id'])
                    connection.register('batch',candidates.to_arrow())
                    query = ('SELECT b.split,b.feed_id,b.candidate_index,b.item_id,'
                             + ','.join(feature_names())
                             + ' FROM batch b LEFT JOIN item_lookup USING (item_id) '
                             'LEFT JOIN category_lookup USING (category_id) ORDER BY b.feed_id,b.candidate_index')
                    features = pl.from_arrow(connection.execute(query).fetch_arrow_table())
                    connection.unregister('batch')
                    validate_features(features,entry['candidates'])
                    if not features.select(*KEY,'item_id').equals(candidates.sort('feed_id','candidate_index').select(*KEY,'item_id')):
                        raise ValueError('itemstats candidate keys changed')
                    write_parquet_atomic(features,path)
                saved[entry['source']] = {'path':path.relative_to(output).as_posix(),'rows':entry['candidates'],
                    'source':stamp,'fingerprint':fingerprint(path),
                    'duration_seconds':old.get('duration_seconds',0.) if reuse else round(time.monotonic()-part_started,3)}
                manifest['splits'][split] = {'parts':saved,'covered_parts':len(saved),
                                             'total_parts':len(source['splits'][split]['parts'])}
                manifest['stage'] = f'features_{split}'
                manifest['resources'] = memory_snapshot()
                write_json(manifest_path,manifest)
                if not reuse or number == len(entries):
                    print(f'itemstats {split}: {number}/{len(entries)} shards; {time.monotonic()-started:.0f}s',flush=True)
        for relative,stamp in inputs.items():
            if fingerprint(prepared/relative) != stamp:
                raise ValueError('source changed during itemstats generation')
        manifest.update(complete=True,stage='complete',all_train_parts=False,
                        duration_seconds=round(time.monotonic()-started,2),resources=memory_snapshot())
        write_json(manifest_path,manifest)
        return manifest
    finally:
        connection.close()


def validate_features(frame, expected_rows):
    schema = {'split':pl.String,'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
              **{name:pl.Float32 for name in feature_names()}}
    if frame.height != expected_rows or frame.schema != schema or frame.select(KEY).is_duplicated().any():
        raise ValueError('invalid itemstats schema, count or candidate keys')
    if frame.null_count().to_numpy().any():
        raise ValueError('null itemstats feature')
    for name in feature_names():
        if not frame[name].is_finite().all() or (frame[name] < 0).any():
            raise ValueError(f'invalid itemstats feature: {name}')
    for entity in ('item','category'):
        if (frame[f'is_{entity}_ctr'] > 1).any():
            raise ValueError('itemstats CTR exceeds one')
