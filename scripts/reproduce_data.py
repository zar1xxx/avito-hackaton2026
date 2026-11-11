"""One command after downloading/extracting the competition archive.

Creates candidate/feed tables, full EDA, validation assignments, versioned item
catalogue and cosine features. All generated data remain outside Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path('runs/.matplotlib').resolve()))

import polars as pl
import pyarrow.parquet as pq

from scripts.build_eda_notebook import build as build_notebook
from src.common.data import list_parts
from src.common.embedding_catalogue import build_catalogue
from src.common.preparation import FORMAT_VERSION, fingerprint, prepare_dataset
from src.common.similarity import build_features
from src.common.validation import make_validation_split


def prepared_cache_valid(raw, prepared, eda):
    """Trust a complete cache only after checking source/output identities and counts."""
    try:
        manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
        summary = eda / 'summary.json'
        if not manifest.get('complete') or manifest['format_version'] != FORMAT_VERSION:
            return False
        if hashlib.sha256(summary.read_bytes()).hexdigest() != manifest['eda_summary_sha256']:
            return False
        for split, info in manifest['splits'].items():
            for entry in info['parts']:
                source = raw / split / 'impressions' / entry['source']
                state = prepared / '_state' / split / (source.stem + '.json')
                checkpoint = json.loads(state.read_text(encoding='utf-8'))
                if (fingerprint(source) != entry['source_fingerprint'] or checkpoint.get('source') != entry['source_fingerprint']
                        or checkpoint['format_version'] != FORMAT_VERSION):
                    return False
                for kind, key, stamp_key in [('candidates', 'candidate_path', 'candidate_file'), ('feeds', 'feed_path', 'feed_file')]:
                    path = prepared / entry[key]
                    if fingerprint(path) != checkpoint[stamp_key] or pq.read_metadata(path).num_rows != entry[kind]:
                        return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def reproduce(raw=Path('competition'), prepared=Path('data/prepared'), eda=Path('runs/eda'),
              catalogue=Path('data/embedding_catalogue'), features=Path('data/features/cosine'),
              *, resume=False, feature_train_parts=1):
    raw, prepared, eda, catalogue, features = (Path(path).resolve() for path in (raw, prepared, eda, catalogue, features))
    if not (raw / 'train/impressions').is_dir() and (raw / 'competition/train/impressions').is_dir():
        raw = raw / 'competition'
    if resume and prepared_cache_valid(raw, prepared, eda):
        print('Prepared dataset: all source/output identities and row counts verified; cache reused.', flush=True)
    else:
        prepare_dataset(raw, prepared, eda, resume=resume)
    feeds = pl.scan_parquet(list_parts(raw, 'val')).select('feed_id', 'timestamp').collect()
    expected = make_validation_split(feeds, public_fraction=.3, n_folds=5, seed=42)
    path = prepared / 'val_folds.parquet'
    if path.exists():
        if not pl.read_parquet(path).equals(expected):
            raise ValueError('existing validation assignments differ; choose a fresh output directory')
    else:
        expected.write_parquet(path)
    build_notebook(eda / 'summary.json', eda / '01_data_eda.ipynb')
    build_catalogue(raw, catalogue, resume=resume)
    build_features(raw, prepared, catalogue, features, train_parts=feature_train_parts, resume=resume)
    print('Data, EDA, folds, versioned catalogue and cosine features are ready.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--eda', type=Path, default=Path('runs/eda'))
    parser.add_argument('--catalogue', type=Path, default=Path('data/embedding_catalogue'))
    parser.add_argument('--features', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--feature-train-parts', type=int, default=1,
                        help='train cosine cache size (default: the one-shard baseline; val/test are complete)')
    parser.add_argument('--all-train-features', action='store_true',
                        help='materialize cosine features for all 512 train shards; may take hours')
    args = parser.parse_args()
    reproduce(args.raw, args.prepared, args.eda, args.catalogue, args.features,
              resume=args.resume, feature_train_parts=None if args.all_train_features else args.feature_train_parts)
