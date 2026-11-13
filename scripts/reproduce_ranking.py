"""Reproduce selected feature groups, CatBoost predictions and a validated submission."""
import argparse
import hashlib
from pathlib import Path

from scripts.reproduce_data import reproduce
from src.common.data import list_parts


def reproduce_ranking(raw=Path('competition'), output=Path('runs/reproduced-ranking'), *,
                      prepared=Path('data/prepared'), feature_dir=Path('data/features'),
                      catalogue=Path('data/embedding_catalogue'),
                      train_parts=8, infeed=False, history=False, itemstats=False, userstats=False,
                      iterations=100, depth=6, seed=42, threads=2, memory_mb=2048, resume=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a fresh output directory for the model run')
    if train_parts < 1 or iterations < 1 or depth < 1 or threads < 1 or memory_mb < 128:
        raise ValueError('invalid training or resource parameters')
    raw, prepared, feature_dir = (Path(path).resolve() for path in (raw, prepared, feature_dir))
    if not (raw / 'train/impressions').is_dir() and (raw / 'competition/train/impressions').is_dir():
        raw = raw / 'competition'
    available_parts = len(list_parts(raw, 'train'))
    if train_parts > available_parts:
        raise ValueError('requested training size exceeds the source shard count')
    if (itemstats or userstats) and train_parts == available_parts:
        raise ValueError('statistics require at least one donor shard outside model training')
    reproduce(raw, prepared, output.parent / 'eda', catalogue=Path(catalogue), features=feature_dir / 'cosine',
              feature_train_parts=train_parts, infeed=feature_dir / 'infeed' if infeed else None,
              resume=resume)
    roots = [feature_dir / 'cosine']
    if infeed:
        roots.append(feature_dir / 'infeed')
    if history:
        from src.common.history import build_features
        root = feature_dir / 'history'
        build_features(prepared, root, train_parts=train_parts, resume=resume)
        roots.append(root)
    for enabled, name in ((itemstats, 'itemstats'), (userstats, 'userstats')):
        if enabled:
            if name == 'itemstats':
                from src.common.itemstats import build_features
            else:
                from src.common.userstats import build_features
            root = feature_dir / f'{name}-{train_parts:03d}'
            build_features(prepared, root, train_parts=train_parts, memory_mb=memory_mb,
                           threads=1, resume=resume)
            roots.append(root)
    from scripts.run_baseline_check import run
    report = run(prepared, raw, output, train_parts=train_parts, iterations=iterations,
                 depth=depth, seed=seed, threads=threads, feature_root=roots)
    import polars as pl
    from src.common.preparation import write_json, write_parquet_atomic
    handoff = {}
    for source, name, columns in (
        ('validation_predictions.parquet', 'val_pred.parquet', ['feed_id', 'item_id', 'score', 'target']),
        ('submission.parquet', 'test_pred.parquet', ['feed_id', 'item_id', 'score']),
    ):
        frame = pl.read_parquet(output / source, columns=columns)
        path = output / name
        write_parquet_atomic(frame, path)
        handoff[name] = {'rows': frame.height, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(output / 'pipeline.json', {
        'complete': True, 'pipeline_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_roots': [str(root) for root in roots], 'train_parts': train_parts,
        'model_git': report['git'], 'predictions': handoff,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features'))
    parser.add_argument('--catalogue', type=Path, default=Path('data/embedding_catalogue'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-parts', type=int, default=8)
    parser.add_argument('--infeed', action='store_true')
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--itemstats', action='store_true')
    parser.add_argument('--userstats', action='store_true')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-mb', type=int, default=2048, help='DuckDB aggregation memory limit')
    parser.add_argument('--resume', action='store_true', help='reuse matching data and feature caches')
    args = parser.parse_args()
    reproduce_ranking(args.raw, args.out, prepared=args.prepared, feature_dir=args.feature_dir,
                      catalogue=args.catalogue,
                      train_parts=args.train_parts, infeed=args.infeed, history=args.history,
                      itemstats=args.itemstats, userstats=args.userstats, iterations=args.iterations,
                      depth=args.depth, seed=args.seed, threads=args.threads,
                      memory_mb=args.memory_mb, resume=args.resume)


if __name__ == '__main__':
    main()
