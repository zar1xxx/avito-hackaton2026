"""Small CatBoost run proving the prepared-data -> validation -> submission path.

The checkpoint is selected on pseudo-Public only. All val feeds are scored,
including untouched pseudo-Private, and metrics are checked against the supplied
evaluator. Base inputs are f1-f5 and category_id; optional occurrence-keyed
feature groups are attached explicitly and recorded in the run metadata.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from scripts.validate_submission import validate
from src.common.metric import all_variants
from src.common.preparation import scan_candidates, write_json, write_parquet_atomic
from src.common.resources import memory_snapshot


FEATURES = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id']
REFERENCE = Path('competition/baseline/calculate_metrics.py')


def load_validation(root: Path) -> pl.DataFrame:
    columns = ['feed_id', 'timestamp', 'candidate_index', 'item_id', 'target', *FEATURES]
    frame = scan_candidates(root, 'val').select(columns).collect()
    folds = pl.read_parquet(root / 'val_folds.parquet')
    if not folds.schema['feed_id'].is_integer():
        raise ValueError('validation feed_id must have an integer dtype')
    folds = folds.with_columns(pl.col('feed_id').cast(frame.schema['feed_id']))
    if folds['feed_id'].is_duplicated().any() or folds['feed_id'].null_count():
        raise ValueError('validation assignments must have one non-null row per feed')
    if set(folds['leaderboard_split'].to_list()) != {'public', 'private'}:
        raise ValueError('both pseudo-Public and pseudo-Private must be present')
    if folds.filter(pl.col('leaderboard_split') == 'public')['timestamp'].max() >= folds.filter(pl.col('leaderboard_split') == 'private')['timestamp'].min():
        raise ValueError('pseudo-Public must strictly precede pseudo-Private in time')
    frame = frame.join(folds.select('feed_id', 'leaderboard_split', pl.col('timestamp').alias('fold_timestamp')),
                       on='feed_id', how='left', validate='m:1')
    if frame['leaderboard_split'].null_count() or frame['feed_id'].n_unique() != folds.height:
        raise ValueError('validation assignments do not exactly cover val feeds')
    if frame.filter(pl.col('timestamp') != pl.col('fold_timestamp')).height:
        raise ValueError('assignment timestamps differ from prepared val')
    return frame.sort('feed_id', 'candidate_index')


def pool(frame: pl.DataFrame, labeled=True, features=None) -> Pool:
    return Pool(frame.select(features or FEATURES).to_pandas(),
                label=frame['target'].to_numpy() if labeled else None,
                group_id=frame['feed_id'].to_numpy(), cat_features=['category_id'])


def git_snapshot():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def run(root: Path, raw: Path, output: Path, *, train_parts=1, iterations=100,
        depth=6, seed=42, threads=2, reference_path=REFERENCE, feature_root=None):
    root, raw, output = (Path(path).resolve() for path in (root, raw, output))
    if output.exists():
        raise FileExistsError(f'choose a new run directory: {output}')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('prepared data are incomplete')
    parts = manifest['splits']['train']['parts']
    if not 1 <= train_parts <= len(parts) or iterations < 1 or depth < 1 or threads < 1:
        raise ValueError('invalid training size or model parameters')
    reference_path = Path(reference_path).resolve()
    spec = importlib.util.spec_from_file_location('supplied_evaluator', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    selected = [parts[int(index)] for index in np.linspace(0, len(parts) - 1, train_parts, dtype=int)]
    started = time.monotonic()
    resources={'start':memory_snapshot()}
    code_root = Path(__file__).resolve().parents[1]
    tracked_code = ['scripts/run_baseline_check.py', 'scripts/validate_submission.py',
                    'src/common/preparation.py', 'src/common/metric.py', 'src/common/resources.py']
    model_features = list(FEATURES)
    feature_metadata = []
    feature_roots = [] if feature_root is None else ([feature_root] if isinstance(feature_root, (str, Path)) else list(feature_root))
    feature_roots = [Path(path).resolve() for path in feature_roots]
    for group_root in feature_roots:
        from src.common.similarity import attach_features
        feature_manifest_path = group_root / 'manifest.json'
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding='utf-8'))
        if not feature_manifest.get('complete'):
            raise ValueError(f'feature group is incomplete: {group_root}')
        overlap = set(model_features) & set(feature_manifest['features'])
        if overlap:
            raise ValueError(f'duplicate feature names across groups: {sorted(overlap)}')
        model_features += feature_manifest['features']
        tracked_code += ['src/common/similarity.py', 'src/common/embedding_catalogue.py']
        if feature_manifest.get('feature_group') == 'infeed':
            tracked_code.append('src/common/infeed.py')
        if feature_manifest.get('feature_group') == 'itemstats':
            tracked_code.append('src/common/itemstats.py')
        feature_metadata.append({'group': feature_manifest.get('feature_group', 'cosine'),
                                 'manifest_sha256': hashlib.sha256(feature_manifest_path.read_bytes()).hexdigest(),
                                 'features': feature_manifest['features'],
                                 'vector_policy': feature_manifest.get('vector_policy'),
                                 'covered_parts': {name: value['covered_parts'] for name,value in feature_manifest['splits'].items()}})
    report = {
        'complete': False, 'stage':'loading_data', 'purpose': 'infrastructure baseline, small training subset',
        'git': git_snapshot(), 'code_sha256': {name: hashlib.sha256((code_root / name).read_bytes()).hexdigest() for name in dict.fromkeys(tracked_code)},
        'prepared_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'features': model_features, 'resources':resources,
        'similarity': next((group for group in feature_metadata if group['group'] == 'cosine'), None),
        'feature_groups': feature_metadata, 'train_parts': [part['source'] for part in selected],
        'selection': 'best NDCG@30 on pseudo-Public; pseudo-Private never passed to fit',
        'score_dtype': 'float32', 'score_ties': 'unmodified model scores; pessimistic reference tie handling',
        'parameters': {'iterations': iterations, 'depth': depth, 'seed': seed, 'threads': threads,
                       'learning_rate': .1, 'loss_function': 'YetiRank'},
    }
    output.mkdir(parents=True)
    write_json(output / 'run.json', report)
    columns = ['feed_id', 'candidate_index', 'item_id', 'target', *FEATURES]
    train = pl.concat([pl.read_parquet(root / part['candidate_path'], columns=columns) for part in selected])\
        .sort('feed_id', 'candidate_index')
    val = load_validation(root)
    for group_root in feature_roots:
        train, _ = attach_features(train, group_root, 'train', [entry['source'] for entry in selected])
        val, _ = attach_features(val, group_root, 'val', [entry['source'] for entry in manifest['splits']['val']['parts']])
    if feature_roots:
        train, val = train.sort('feed_id', 'candidate_index'), val.sort('feed_id', 'candidate_index')
    if val.height != manifest['splits']['val']['candidates']:
        raise ValueError('val candidate count differs from prepared manifest')
    public = val.filter(pl.col('leaderboard_split') == 'public')
    report['counts'] = {'train_rows': train.height, 'train_feeds': train['feed_id'].n_unique(),
                        'val_rows': val.height, 'val_feeds': val['feed_id'].n_unique(),
                        'selection_rows': public.height, 'selection_feeds': public['feed_id'].n_unique(),
                        'train_fraction_of_candidates': train.height / manifest['splits']['train']['candidates']}
    print(f"Training on {train.height:,} candidates / {train['feed_id'].n_unique():,} feeds; "
          f"checkpoint selection on {public['feed_id'].n_unique():,} pseudo-Public feeds", flush=True)
    model = CatBoostRanker(loss_function='YetiRank', iterations=iterations, depth=depth,
                          learning_rate=.1, eval_metric='NDCG:top=30;type=Base',
                          random_seed=seed, thread_count=threads, allow_writing_files=False,
                          used_ram_limit='1024mb')
    train_pool, public_pool = pool(train, features=model_features), pool(public, features=model_features)
    resources['pools_loaded']=memory_snapshot()
    report['stage']='fit'
    write_json(output/'run.json',report)
    model.fit(train_pool, eval_set=public_pool, use_best_model=True,
              early_stopping_rounds=25, verbose=25)
    model.save_model(str(output / 'model.cbm'))
    report['trees'] = model.tree_count_
    report['model_parameters'] = model.get_all_params()
    resources['fit_complete']=memory_snapshot()
    report['stage']='validation'
    write_json(output/'run.json',report)
    del train_pool, public_pool, train, public
    gc.collect()
    scores = model.predict(pool(val, labeled=False, features=model_features), thread_count=threads).astype(np.float32)
    scored = val.select('feed_id', 'candidate_index', 'item_id', 'target', 'leaderboard_split')\
        .with_columns(pl.Series('score', scores, dtype=pl.Float32))
    write_parquet_atomic(scored, output / 'validation_predictions.parquet')
    report['validation'] = {}
    for name, frame in [('all', scored), ('pseudo_public', scored.filter(pl.col('leaderboard_split') == 'public')),
                        ('pseudo_private', scored.filter(pl.col('leaderboard_split') == 'private'))]:
        metrics = all_variants(frame['feed_id'], frame['target'], frame['score'])
        official = reference.metrics(frame)
        for ours, theirs in [('ndcg_full', 'ndcg'), ('ndcg@10', 'ndcg_at_10'), ('ndcg@30', 'ndcg_at_30')]:
            if abs(metrics[ours] - official[theirs]) >= 1e-10:
                raise ValueError(f'{name}: metric mismatch for {ours}')
        report['validation'][name] = {'feeds': frame['feed_id'].n_unique(), 'rows': frame.height,
                                     **metrics, 'reference_check': 'passed'}
        print(name, report['validation'][name], flush=True)
    del val, scored, scores
    gc.collect()
    # Check the serialized model path too; inference happens per prepared shard.
    reloaded = CatBoostRanker()
    reloaded.load_model(str(output / 'model.cbm'))
    predictions = []
    for index, part in enumerate(manifest['splits']['test']['parts']):
        batch = pl.read_parquet(root / part['candidate_path'], columns=['feed_id', 'candidate_index', 'item_id', *FEATURES])
        for group_root in feature_roots:
            batch, _ = attach_features(batch, group_root, 'test', [part['source']])
        score = reloaded.predict(pool(batch, labeled=False, features=model_features), thread_count=threads).astype(np.float32)
        predictions.append(batch.select('feed_id', 'item_id').with_columns(pl.Series('score', score, dtype=pl.Float32)))
        if (index + 1) % 16 == 0:
            print(f"test: {index + 1}/{len(manifest['splits']['test']['parts'])} shards", flush=True)
    submission = pl.concat(predictions)
    if submission.height != manifest['splits']['test']['candidates']:
        raise ValueError('test candidate count differs from prepared manifest')
    submission_path = output / 'submission.parquet'
    write_parquet_atomic(submission, submission_path)
    errors = validate(str(raw / 'sample_submission.parquet'), str(submission_path))
    if errors:
        raise ValueError(f'submission validation failed: {errors}')
    report['submission'] = {'rows': submission.height, 'feeds': submission['feed_id'].n_unique(),
                            'validator': 'passed', 'path': submission_path.name,
                            'sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
                            'uploaded_to_competition': False}
    report['duration_seconds'] = round(time.monotonic() - started, 2)
    resources['finished']=memory_snapshot()
    report['complete'] = True
    report['stage']='complete'
    write_json(output / 'run.json', report)
    print(f"Complete: {output}; submission validator passed", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--feature-root', type=Path, action='append', help='attach a feature group; repeat for multiple groups')
    args = parser.parse_args()
    run(args.data, args.raw, args.out, train_parts=args.train_parts, iterations=args.iterations,
        depth=args.depth, seed=args.seed, threads=args.threads, reference_path=args.reference, feature_root=args.feature_root)


if __name__ == '__main__':
    main()
