"""Prepare candidate/feed Parquet datasets and compute EDA in one bounded-memory pass."""
import argparse
from pathlib import Path

from src.common.preparation import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("competition"))
    parser.add_argument("--out", type=Path, default=Path("data/prepared"))
    parser.add_argument("--eda-out", type=Path, default=Path("runs/eda"))
    parser.add_argument("--resume", action="store_true", help="reuse unchanged, verified output shards")
    args = parser.parse_args()
    prepare_dataset(args.data, args.out, args.eda_out, resume=args.resume)


if __name__ == "__main__":
    main()
