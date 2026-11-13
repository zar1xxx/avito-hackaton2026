import argparse
from pathlib import Path

from src.common.infeed import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build rank/z-score, size and category-share features for complete feeds.')
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--out', type=Path, default=Path('data/features/infeed'))
    parser.add_argument('--train-parts', type=int, default=None, help='optional subset; default covers the entire train')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_features(args.prepared, args.out, train_parts=args.train_parts, resume=args.resume)
