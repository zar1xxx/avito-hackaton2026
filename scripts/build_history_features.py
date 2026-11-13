import argparse
from pathlib import Path

from src.common.history import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build candidate-vs-user-history features (item/category/event type).')
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, default=Path('data/features/history'))
    parser.add_argument('--train-parts', type=int, default=None, help='optional subset; default covers the entire train')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.out, train_parts=args.train_parts, resume=args.resume)
