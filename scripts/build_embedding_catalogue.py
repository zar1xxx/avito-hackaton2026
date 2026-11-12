import argparse
from pathlib import Path

from src.common.embedding_catalogue import build_catalogue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and audit a split-versioned common item embedding catalogue.')
    parser.add_argument('--data', type=Path, default=Path('competition'))
    parser.add_argument('--out', type=Path, default=Path('data/embedding_catalogue'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    build_catalogue(args.data, args.out, resume=args.resume)
