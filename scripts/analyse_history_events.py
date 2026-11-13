"""Measure target rates conditional on item or category history event types."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.history import EVENT_TYPES, KEY
from src.common.preparation import fingerprint, write_json


def analyse(prepared, cache, output, *, train_parts=8, entity='item'):
    if entity not in ('item', 'category'):
        raise ValueError('entity must be item or category')
    prefix = 'h_item_evt' if entity == 'item' else 'h_cat_evt'
    prepared,cache,output = map(Path,(prepared,cache,output))
    source_bytes = (prepared/'manifest.json').read_bytes()
    source = json.loads(source_bytes)
    manifest_bytes = (cache/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if not source.get('complete') or not manifest.get('complete'):
        raise ValueError('complete prepared tables and history cache required')
    if manifest.get('source_manifest_sha256') != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError('prepared manifest differs from history source')
    train = source['splits']['train']['parts']
    if not 1 <= train_parts <= len(train):
        raise ValueError('train_parts outside source range')
    selected = [train[int(index)] for index in np.linspace(0,len(train)-1,train_parts,dtype=int)]
    report = {'complete':False,'entity':entity,'train_parts':[entry['source'] for entry in selected],
              'history_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
              'prepared_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),
              'condition':f'{prefix}N > 0; each candidate occurrence counted once per event type',
              'interpretation':'descriptive association; event groups overlap; category mix, history length and train position affect rates',
              'splits':{}}
    event_columns = [f'{prefix}{event}' for event in EVENT_TYPES]
    counter_fields = ['rows', 'positives', 'feeds']
    if entity == 'item':
        counter_fields.append('matching_events')
    for split,entries in [('train',selected),('val',source['splits']['val']['parts'])]:
        total = {'rows':0,'positives':0,'feeds':0}
        counters = {event:dict.fromkeys(counter_fields, 0) for event in EVENT_TYPES}
        for number,entry in enumerate(entries,1):
            part = manifest['splits'][split]['parts'].get(entry['source'])
            if part is None:
                raise ValueError(f'history cache does not cover requested {split} shard')
            path = cache/part['path']
            if (fingerprint(path) != part['fingerprint']
                    or fingerprint(prepared/entry['candidate_path']) != part['source']
                    or fingerprint(prepared/entry['feed_path']) != part['feeds_source']):
                raise ValueError('changed feature or source file')
            keys = [*KEY,'item_id']
            labels = pl.read_parquet(prepared/entry['candidate_path'],columns=[*keys,'target'])
            features = pl.read_parquet(path,columns=[*keys,*event_columns])
            frame = labels.join(features,on=keys,how='left',validate='1:1')
            if (features.height != labels.height or frame.null_count().to_numpy().any()
                    or not labels['target'].is_in([0,1]).all()):
                raise ValueError('invalid labels or incomplete candidate coverage')
            for column in event_columns:
                if not frame[column].is_finite().all() or (frame[column] < 0).any():
                    raise ValueError('invalid event count')
                if entity == 'category' and (frame[column] > 1).any():
                    raise ValueError('category event share exceeds one')
            total['rows'] += frame.height
            total['positives'] += int(frame['target'].sum())
            total['feeds'] += frame['feed_id'].n_unique()
            expressions = []
            for event in EVENT_TYPES:
                column = pl.col(f'{prefix}{event}')
                present = column > 0
                expressions.extend([present.sum().alias(f'rows_{event}'),
                                    pl.col('target').filter(present).sum().alias(f'positives_{event}'),
                                    pl.col('feed_id').filter(present).n_unique().alias(f'feeds_{event}')])
                if entity == 'item':
                    expressions.append(column.sum().alias(f'matching_events_{event}'))
            counts = frame.select(expressions).row(0,named=True)
            for event,values in counters.items():
                for field in values:
                    values[field] += int(counts[f'{field}_{event}'])
            if number == 1 or number % 8 == 0 or number == len(entries):
                print(f'event analysis {split}: {number}/{len(entries)}',flush=True)
        total['target_rate'] = total['positives']/total['rows'] if total['rows'] else None
        events = []
        for event,values in counters.items():
            rows,positives = values['rows'],values['positives']
            rate = positives/rows if rows else None
            absent_rows = total['rows']-rows
            events.append({'event_type':event,**values,'target_rate':rate,
                           'candidate_share':rows/total['rows'],
                           'rate_ratio_to_baseline':rate/total['target_rate'] if rate is not None and total['target_rate'] else None,
                           'absent_target_rate':(total['positives']-positives)/absent_rows if absent_rows else None})
        report['splits'][split] = {'baseline':total,'events':events}
    if ((cache/'manifest.json').read_bytes() != manifest_bytes
            or (prepared/'manifest.json').read_bytes() != source_bytes):
        raise ValueError('history cache or prepared manifest changed during analysis')
    report['complete'] = True
    report['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output.with_suffix('.json'),report)
    condition = ('кандидат встречался в истории с данным типом события'
                 if entity == 'item' else 'категория кандидата встречалась в истории с данным типом события')
    lines = ['# Типы событий и доля целевых действий','',
             f'Условие — {condition} (`{prefix}N > 0`). '
             'Группы пересекаются. Каждое вхождение кандидата учитывается один раз внутри группы. '
             'На доли влияют состав категорий, длина истории и позиция показа в train. '
             'Таблица описывает статистическую связь с target.','',
             f'Train содержит {train_parts} выбранных шардов. Val проверен полностью.','',
             '| Выборка | Тип события | Кандидатов | Позитивов | Лент | Доля target=1 | Отношение к общей доле |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split,group in report['splits'].items():
        baseline = group['baseline']
        lines.append(f"| {split} | Все кандидаты | {baseline['rows']} | {baseline['positives']} | {baseline['feeds']} | {baseline['target_rate']:.6f} | 1.000 |")
        for event in group['events']:
            rate = '—' if event['target_rate'] is None else f"{event['target_rate']:.6f}"
            ratio = '—' if event['rate_ratio_to_baseline'] is None else f"{event['rate_ratio_to_baseline']:.3f}"
            lines.append(f"| {split} | {event['event_type']} | {event['rows']} | {event['positives']} | {event['feeds']} | {rate} | {ratio} |")
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/history'))
    parser.add_argument('--out',type=Path,help='report prefix; default includes entity and train shard count')
    parser.add_argument('--train-parts',type=int,default=8)
    parser.add_argument('--entity',choices=['item','category'],default='item')
    args = parser.parse_args()
    tag = 'category_' if args.entity == 'category' else ''
    output = args.out or Path(f'experiments/history_{tag}event_target_{args.train_parts:03d}')
    analyse(args.prepared,args.cache,output,train_parts=args.train_parts,entity=args.entity)
