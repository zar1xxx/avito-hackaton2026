import argparse
from pathlib import Path

from src.common.similarity import build_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build split-safe batched cosine features keyed by candidate occurrence.')
    parser.add_argument('--raw', type=Path, default=Path('competition'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--catalogue', type=Path, default=Path('data/embedding_catalogue'))
    parser.add_argument('--out', type=Path, default=Path('data/features/cosine'))
    parser.add_argument('--train-parts', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reader',choices=['auto','mapped','sequential'],default='auto')
    args = parser.parse_args()
    build_features(args.raw, args.prepared, args.catalogue, args.out,
                   train_parts=args.train_parts, batch_size=args.batch_size, resume=args.resume,reader=args.reader)
