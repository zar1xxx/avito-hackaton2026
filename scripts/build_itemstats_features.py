"""Build item/category statistics with train-shard exclusion and bounded RAM."""
import argparse
from pathlib import Path

from src.common.itemstats import build_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--out',type=Path,default=Path('data/features/itemstats-008'))
    parser.add_argument('--train-parts',type=int,default=8)
    parser.add_argument('--smoothing',type=float,default=20.)
    parser.add_argument('--memory-mb',type=int,default=1024)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    build_features(args.prepared,args.out,train_parts=args.train_parts,smoothing=args.smoothing,
                   memory_mb=args.memory_mb,threads=args.threads,resume=args.resume)
