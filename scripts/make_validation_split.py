"""python -m scripts.make_validation_split --data competition --out runs/val_folds.parquet"""
import argparse
import json
from pathlib import Path

import polars as pl

from src.common.data import list_parts
from src.common.validation import make_validation_split


def main():
    parser = argparse.ArgumentParser(description="Create time holdout and grouped folds for val.")
    parser.add_argument("--data", type=Path, default=Path("competition"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-fraction", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    frame = pl.scan_parquet(list_parts(args.data, "val")).select("feed_id", "timestamp").collect()
    result = make_validation_split(frame, args.public_fraction, args.folds, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.out)
    print(json.dumps({"output": str(args.out), "feeds": result.height,
                      "public_feeds": result.filter(pl.col("leaderboard_split") == "public").height,
                      "private_feeds": result.filter(pl.col("leaderboard_split") == "private").height,
                      "folds": args.folds, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
