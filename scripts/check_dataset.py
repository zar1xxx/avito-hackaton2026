"""Check archive layout, shard schemas, all feed IDs and sampled feed contents.

Run: python -m scripts.check_dataset --data competition --sample-feeds 100
The report states sampling explicitly; this is not a full EDA scan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.validate_submission import validate
from src.common.data import CAND_COLS, EMB_DIM


HISTORY_COLS = [f"history__{name}" for name in
                ("item_id", "category_id", "event_type", "event_source", "item_id__embedding_idx")]


def inspect_split(root: Path, split: str, sample_feeds: int) -> dict:
    errors = []
    report = {"errors": errors, "sampled_feeds": 0}
    parts = sorted((root / split / "impressions").glob("part-*.parquet"))
    if not parts:
        errors.append(f"missing {split}/impressions/part-*.parquet")
        return report
    embedding_dir = root / split / "item_embeddings"
    try:
        meta = json.loads((embedding_dir / "data.meta").read_text(encoding="utf-8"))
        shape = meta["shape"]
        dtype = np.dtype(meta["dtype"])
        if len(shape) != 2 or any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError("invalid embedding shape")
        if shape[1] != EMB_DIM or dtype != np.dtype("int8") or meta["order"] not in ("C", "F"):
            raise ValueError("expected int8 embeddings with 128 dimensions and C/F order")
        expected_size = int(np.prod(shape)) * dtype.itemsize
        actual_size = (embedding_dir / "data.memmap").stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"memmap size {actual_size}, expected {expected_size}")
        report["embedding_shape"] = shape
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"embeddings: {exc}")
        return report
    arrays = CAND_COLS + (["target"] if split != "test" else []) + (["pos"] if split == "train" else [])
    required = ["feed_id", "timestamp", "user_id", "user_embedding"] + arrays + HISTORY_COLS
    samples = []
    remaining = sample_feeds
    reference_schema = None
    for part in parts:
        try:
            schema = pl.read_parquet_schema(part)
            missing = sorted(set(required) - set(schema))
            if missing:
                errors.append(f"{part.name}: missing columns {missing}")
                continue
            if not schema["feed_id"].is_integer() or not schema["user_id"].is_integer():
                errors.append(f"{part.name}: feed_id and user_id must be integers")
            for name in arrays + HISTORY_COLS + ["user_embedding"]:
                if not isinstance(schema[name], pl.List):
                    errors.append(f"{part.name}: {name} must be a list")
            selected_schema = {name: schema[name] for name in required}
            if reference_schema is not None and selected_schema != reference_schema:
                errors.append(f"{part.name}: schema differs from the first shard")
            reference_schema = selected_schema
            if remaining > 0:
                batch = pl.read_parquet(part, columns=required, n_rows=remaining)
                samples.extend(batch.iter_rows(named=True))
                remaining -= batch.height
        except (OSError, pl.exceptions.PolarsError) as exc:
            errors.append(f"{part.name}: {exc}")
    report.update(parts=len(parts), sampled_feeds=len(samples))
    if errors:
        return report
    # ID projection covers every feed; candidate arrays below are sampled.
    try:
        counts = pl.scan_parquet(parts).select(
            pl.len().alias("feeds"), pl.col("feed_id").n_unique().alias("unique_feeds"),
            pl.col("feed_id").null_count().alias("null_feed_ids"),
            pl.col("timestamp").null_count().alias("null_timestamps"),
        ).collect().row(0, named=True)
        report.update(counts)
        if counts["null_feed_ids"] or counts["null_timestamps"]:
            errors.append("null feed_id or timestamp")
        if counts["feeds"] != counts["unique_feeds"]:
            errors.append("duplicate feed_id across impression rows")
        if not counts["feeds"]:
            errors.append("empty split")
    except pl.exceptions.PolarsError as exc:
        errors.append(f"feed ID scan: {exc}")
    for row in samples:
        prefix = f"feed {row['feed_id']}"
        if any(row[name] is None for name in required):
            errors.append(f"{prefix}: null scalar or list")
            continue
        for columns in (arrays, HISTORY_COLS):
            if len({len(row[name]) for name in columns}) != 1:
                errors.append(f"{prefix}: mismatched list lengths: {columns}")
        for name in arrays + HISTORY_COLS + ["user_embedding"]:
            if any(value is None for value in row[name]):
                errors.append(f"{prefix}: null inside {name}")
        if not row["item_id"] or len(set(row["item_id"])) != len(row["item_id"]):
            errors.append(f"{prefix}: empty or duplicate candidate item_id")
        if len(row["user_embedding"]) != EMB_DIM:
            errors.append(f"{prefix}: wrong user_embedding dimension")
        for name in ("item_id__embedding_idx", "history__item_id__embedding_idx"):
            if any(i is not None and (i < -1 or i >= shape[0]) for i in row[name]):
                errors.append(f"{prefix}: {name} outside [-1, embedding_rows)")
        if "target" in row and any(t not in (0, 1) for t in row["target"]):
            errors.append(f"{prefix}: nonbinary target")
    return report


def inspect_dataset(root: Path, sample_feeds: int = 100) -> dict:
    if sample_feeds < 1:
        raise ValueError("sample_feeds must be positive")
    result = {"data": str(root), "sample_limit_per_split": sample_feeds,
              "splits": {name: inspect_split(root, name, sample_feeds) for name in ("train", "val", "test")}}
    template = root / "sample_submission.parquet"
    if template.is_file():
        result["template_errors"] = validate(str(template), str(template))
    else:
        result["template_errors"] = ["missing sample_submission.parquet"]
    result["ok"] = not result["template_errors"] and not any(
        value["errors"] for value in result["splits"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("competition"))
    parser.add_argument("--sample-feeds", type=int, default=100)
    args = parser.parse_args()
    if args.sample_feeds < 1:
        parser.error("--sample-feeds must be positive")
    result = inspect_dataset(args.data, args.sample_feeds)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
