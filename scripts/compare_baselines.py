"""Paired comparison on exactly the same val rows and temporal assignments."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.metric import ndcg_one
from src.common.preparation import write_json


def compare(baseline, candidate, output, *, replicates=1000, seed=42):
    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    a, b = [json.loads((path / 'run.json').read_text(encoding='utf-8')) for path in (baseline, candidate)]
    if not a.get('complete') or not b.get('complete') or replicates < 10:
        raise ValueError('complete runs and at least ten bootstrap replicates are required')
    for key in ('prepared_manifest_sha256', 'train_parts', 'parameters'):
        if a[key] != b[key]:
            raise ValueError(f'unmatched experiment setup: {key}')
    left, right = [pl.read_parquet(path / 'validation_predictions.parquet') for path in (baseline, candidate)]
    keys = ['feed_id', 'candidate_index', 'item_id']
    joined = left.join(right, on=keys, how='inner', suffix='_new', validate='1:1').sort('feed_id', 'candidate_index')
    if joined.height != left.height or joined.height != right.height:
        raise ValueError('different validation candidate keys')
    if joined.filter((pl.col('target') != pl.col('target_new')) |
                     (pl.col('leaderboard_split') != pl.col('leaderboard_split_new'))).height:
        raise ValueError('different validation targets or holdout assignments')
    ids = joined['feed_id'].to_numpy()
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    target, old_score, new_score = [joined[name].to_numpy() for name in ('target', 'score', 'score_new')]
    part = joined['leaderboard_split'].to_numpy()
    by_split = {'public': [], 'private': []}
    for start, end in zip(bounds[:-1], bounds[1:]):
        old = ndcg_one(target[start:end], old_score[start:end], cutoff=30)
        new = ndcg_one(target[start:end], new_score[start:end], cutoff=30)
        by_split[part[start]].append((old, new))
    result = {'matched_setup': True, 'val_rows': joined.height, 'train_parts': a['train_parts'],
              'baseline_commit': a.get('git', {}).get('commit'), 'candidate_commit': b.get('git', {}).get('commit'),
              'selection': 'both checkpoints selected on pseudo-Public; fixed pseudo-Private compared after fitting',
              'scores_dtype': 'float32', 'metrics': {}}
    for name, pairs in [*by_split.items(), ('all', by_split['public'] + by_split['private'])]:
        if not pairs:
            raise ValueError('both validation holdouts must be nonempty')
        values = np.array(pairs)
        old, new = values.mean(axis=0)
        report_name = 'all' if name == 'all' else 'pseudo_' + name
        if abs(old - a['validation'][report_name]['ndcg@30']) >= 1e-10 or abs(new - b['validation'][report_name]['ndcg@30']) >= 1e-10:
            raise ValueError('saved predictions and recorded NDCG@30 disagree')
        result['metrics'][report_name] = {'feeds': len(pairs), 'baseline_ndcg30': float(old), 'candidate_ndcg30': float(new),
                                         'delta': float(new - old), 'relative_gain_percent': float(100 * (new - old) / old) if old else None}
    private = np.array(by_split['private'])
    delta = private[:, 1] - private[:, 0]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for start in range(0, replicates, 25):
        end = min(start + 25, replicates)
        draw = rng.integers(0, len(delta), size=(end - start, len(delta)), dtype=np.int32)
        samples[start:end] = delta[draw].mean(axis=1)
    result['paired_bootstrap_private'] = {'unit': 'feed', 'replicates': replicates, 'seed': seed,
                                          'delta_95_ci': np.quantile(samples, [.025, .975]).tolist(),
                                          'fraction_delta_positive': float((samples > 0).mean()),
                                          'scope': 'sampling uncertainty on this fixed val; not an estimate for full-train models'}
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/baseline-check-001'))
    parser.add_argument('--candidate', type=Path, default=Path('runs/baseline-cosine-001'))
    parser.add_argument('--out', type=Path, default=Path('runs/baseline-cosine-001/comparison.json'))
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.out)
