"""Bounded-memory preparation and exact aggregate EDA of the public data.

Candidate tables intentionally contain raw candidate features, not expensive
history similarities. Feed-level histories are stored separately, once per feed.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.common.data import CAND_COLS, EMB_DIM, FLOAT_FEATS, list_parts


HISTORY_COLS = [f"history__{name}" for name in (
    "item_id", "category_id", "event_type", "event_source", "item_id__embedding_idx")]
FORMAT_VERSION = 2
POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


class IdBitmap:
    """Exact positive-int32 ID membership using one bit per ID, not a Python set."""

    def __init__(self, include_zero=False):
        self.bits = np.zeros(0, dtype=np.uint8)
        self.include_zero = include_zero

    def update(self, values):
        values = np.asarray(values)
        if not values.size:
            return
        if values.min() < 0 or values.max() > np.iinfo(np.int32).max:
            raise ValueError("IDs must be nonnegative int32")
        if not self.include_zero:
            values = values[values > 0]  # 0 denotes a missing user/item ID.
        if not values.size:
            return
        required = int(values.max()) // 8 + 1
        if required > len(self.bits):
            self.bits = np.pad(self.bits, (0, required - len(self.bits)))
        # at() handles repeated byte indices; ordinary advanced indexing does not.
        np.bitwise_or.at(self.bits, values // 8, (1 << (values % 8)).astype(np.uint8))

    def count(self):
        return int(POPCOUNT[self.bits].sum(dtype=np.int64))

    def intersection(self, other):
        size = min(len(self.bits), len(other.bits))
        return int(POPCOUNT[self.bits[:size] & other.bits[:size]].sum(dtype=np.int64))


def flat_values(series: pl.Series) -> np.ndarray:
    """Flatten list values without the phantom null produced by exploding []."""
    # flatten() respects a sliced list array's offsets; .values exposes the
    # entire underlying child buffer, including rows outside that slice.
    values = series.rechunk().to_arrow().flatten()
    if values.null_count:
        raise ValueError(f"null elements in {series.name}")
    return values.to_numpy(zero_copy_only=False)


def histogram(values) -> Counter:
    unique, counts = np.unique(values, return_counts=True)
    return Counter({int(key): int(count) for key, count in zip(unique, counts)})


def distribution(counts: Counter) -> dict:
    keys = np.array(sorted(counts), dtype=np.int64)
    weights = np.array([counts[int(key)] for key in keys], dtype=np.int64)
    total = int(weights.sum())
    if not total:
        return {"count": 0}
    cumulative = weights.cumsum()

    def quantile(q):
        index = q * (total - 1)
        low, high = math.floor(index), math.ceil(index)
        a, b = keys[np.searchsorted(cumulative, [low + 1, high + 1])]
        return float(a + (b - a) * (index - low))

    return {"count": total, "min": int(keys[0]), "max": int(keys[-1]),
            "mean": float(np.dot(keys.astype(float), weights) / total),
            **{name: quantile(q) for name, q in (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))},
            "histogram": {str(key): counts[int(key)] for key in keys}}


def validate_raw(frame: pl.DataFrame, split: str, embedding_rows: int):
    candidate_cols = CAND_COLS + (["target"] if split != "test" else []) + (["pos"] if split == "train" else [])
    required = ["feed_id", "timestamp", "user_id", "user_embedding"] + candidate_cols + HISTORY_COLS
    if set(required) - set(frame.columns):
        raise ValueError(f"missing required columns: {set(required) - set(frame.columns)}")
    if not frame.height or frame.select(required).null_count().to_numpy().any():
        raise ValueError("empty shard or null scalar/list")
    if frame['feed_id'].is_duplicated().any():
        raise ValueError("duplicate feed_id within shard")
    lengths = frame["item_id"].list.len().to_numpy()
    history_lengths = frame["history__item_id"].list.len().to_numpy()
    if (lengths == 0).any() or (history_lengths > 256).any():
        raise ValueError("empty candidate list or history longer than 256")
    for columns, expected in ((candidate_cols, lengths), (HISTORY_COLS, history_lengths)):
        for name in columns:
            if not np.array_equal(frame[name].list.len().to_numpy(), expected):
                raise ValueError(f"misaligned list: {name}")
    if not (frame["user_embedding"].list.len().to_numpy() == EMB_DIM).all():
        raise ValueError("wrong user embedding length")
    for name in candidate_cols + HISTORY_COLS + ["user_embedding"]:
        values = flat_values(frame[name])
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"nonfinite values: {name}")
    for name in ("feed_id", "user_id"):
        if frame[name].min() < 0:
            raise ValueError(f"negative ID: {name}")
    for name in ("item_id", "history__item_id"):
        values = flat_values(frame[name])
        if values.size and values.min() < 0:
            raise ValueError(f"negative ID: {name}")
    for name in ("item_id__embedding_idx", "history__item_id__embedding_idx"):
        values = flat_values(frame[name])
        if values.size and (values.min() < -1 or values.max() >= embedding_rows):
            raise ValueError(f"embedding index out of bounds: {name}")
    if "target" in candidate_cols and not np.isin(flat_values(frame["target"]), [0, 1]).all():
        raise ValueError("target must be binary")
    if split == "train" and (flat_values(frame["pos"]) < 0).any():
        raise ValueError("negative position")


def flatten_candidates(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    columns = CAND_COLS + (["target"] if split != "test" else []) + (["pos"] if split == "train" else [])
    indexed = frame.with_columns(pl.int_ranges(0, pl.col('item_id').list.len(), dtype=pl.UInt32).alias('candidate_index'))
    result = indexed.select(["feed_id", "timestamp", "user_id", "candidate_index"] + columns).explode(columns + ['candidate_index'])
    types = {"feed_id": pl.Int32, "timestamp": pl.Int32, "user_id": pl.Int32,
             "item_id": pl.Int32, "item_id__embedding_idx": pl.Int32, "category_id": pl.Int8,
             **{name: pl.Float32 for name in FLOAT_FEATS}}
    if split != "test":
        types["target"] = pl.Int8
    if split == "train":
        types["pos"] = pl.Int16
    return result.cast(types).with_columns(pl.lit(split).alias("split")).select("split", pl.all().exclude("split"))


class SplitStatistics:
    def __init__(self, split):
        self.split = split
        self.feeds = self.candidates = self.history_events = 0
        self.positives = 0 if split != "test" else None
        self.candidate_missing_embeddings = self.history_missing_embeddings = 0
        self.zero_candidate_ids = self.zero_history_ids = self.zero_user_ids = 0
        self.duplicate_item_pairs = self.duplicate_item_feeds = self.duplicate_item_rows_excess = self.conflicting_target_pairs = 0
        self.sizes, self.history_sizes, self.positives_per_feed = Counter(), Counter(), Counter()
        self.by_day, self.positions = {}, {}
        self.categories, self.event_types, self.event_sources = Counter(), Counter(), Counter()
        self.feature_moments = {name: {"n": 0, "sum": 0., "sum_squares": 0., "min": math.inf, "max": -math.inf} for name in FLOAT_FEATS}
        self.users, self.items, self.history_items = (IdBitmap() for _ in range(3))
        self.feed_ids = IdBitmap(include_zero=True)
        self.min_timestamp, self.max_timestamp = None, None

    def update(self, frame):
        sizes = frame["item_id"].list.len().to_numpy()
        hist_sizes = frame["history__item_id"].list.len().to_numpy()
        self.feeds += frame.height
        self.candidates += int(sizes.sum(dtype=np.int64))
        self.history_events += int(hist_sizes.sum(dtype=np.int64))
        self.sizes.update(histogram(sizes))
        self.history_sizes.update(histogram(hist_sizes))
        for column, bitmap in (("feed_id", self.feed_ids), ("user_id", self.users)):
            bitmap.update(frame[column].to_numpy())
        self.zero_user_ids += int((frame["user_id"].to_numpy() == 0).sum())
        for column, bitmap, zero_attr in (("item_id", self.items, "zero_candidate_ids"),
                                          ("history__item_id", self.history_items, "zero_history_ids")):
            values = flat_values(frame[column])
            bitmap.update(values)
            setattr(self, zero_attr, getattr(self, zero_attr) + int((values == 0).sum()))
        self.candidate_missing_embeddings += int((flat_values(frame["item_id__embedding_idx"]) == -1).sum())
        self.history_missing_embeddings += int((flat_values(frame["history__item_id__embedding_idx"]) == -1).sum())
        self.categories.update(histogram(flat_values(frame["category_id"])))
        self.event_types.update(histogram(flat_values(frame["history__event_type"])))
        self.event_sources.update(histogram(flat_values(frame["history__event_source"])))
        for name in FLOAT_FEATS:
            values = flat_values(frame[name]).astype(np.float64)
            moments = self.feature_moments[name]
            moments["n"] += values.size
            moments["sum"] += float(values.sum())
            moments["sum_squares"] += float(np.dot(values, values))
            moments["min"] = min(moments["min"], float(values.min()))
            moments["max"] = max(moments["max"], float(values.max()))
        timestamps = frame["timestamp"].to_numpy()
        low, high = int(timestamps.min()), int(timestamps.max())
        self.min_timestamp = low if self.min_timestamp is None else min(self.min_timestamp, low)
        self.max_timestamp = high if self.max_timestamp is None else max(self.max_timestamp, high)
        positive_counts = frame["target"].list.sum().to_numpy().astype(np.int64) if self.positives is not None else None
        if positive_counts is not None:
            self.positives += int(positive_counts.sum())
            self.positives_per_feed.update(histogram(positive_counts))
        for day in np.unique(timestamps // 86400):
            mask = timestamps // 86400 == day
            name = datetime.fromtimestamp(int(day) * 86400, timezone.utc).date().isoformat()
            bucket = self.by_day.setdefault(name, {"feeds": 0, "candidates": 0, "positives": 0 if positive_counts is not None else None})
            bucket["feeds"] += int(mask.sum())
            bucket["candidates"] += int(sizes[mask].sum(dtype=np.int64))
            if positive_counts is not None:
                bucket["positives"] += int(positive_counts[mask].sum())
        if self.split == "train":
            positions = flat_values(frame["pos"])
            targets = flat_values(frame["target"])
            counts = np.bincount(positions)
            positives = np.bincount(positions, weights=targets, minlength=len(counts))
            for position in np.flatnonzero(counts):
                bucket = self.positions.setdefault(str(position), {"impressions": 0, "positives": 0})
                bucket["impressions"] += int(counts[position])
                bucket["positives"] += int(positives[position])

    def summary(self):
        moments = {}
        for name, values in self.feature_moments.items():
            mean = values["sum"] / values["n"]
            moments[name] = {"mean": mean, "std": math.sqrt(max(0., values["sum_squares"] / values["n"] - mean**2)),
                             "min": values["min"], "max": values["max"]}
        for bucket in self.positions.values():
            bucket["ctr"] = bucket["positives"] / bucket["impressions"]
        return {
            "feeds": self.feeds, "candidates": self.candidates, "history_events": self.history_events,
            "positives": self.positives, "ctr": self.positives / self.candidates if self.positives is not None else None,
            "unique_users": self.users.count(), "unique_candidate_items": self.items.count(),
            "unique_history_items": self.history_items.count(),
            "feeds_without_positives": self.positives_per_feed.get(0, 0) if self.positives is not None else None,
            "candidate_missing_embeddings": self.candidate_missing_embeddings,
            "candidate_missing_embedding_rate": self.candidate_missing_embeddings / self.candidates,
            "history_missing_embeddings": self.history_missing_embeddings,
            "history_missing_embedding_rate": self.history_missing_embeddings / self.history_events if self.history_events else 0.,
            "zero_ids": {"users": self.zero_user_ids, "candidates": self.zero_candidate_ids, "history": self.zero_history_ids},
            "duplicate_item_pairs": self.duplicate_item_pairs, "duplicate_item_feeds": self.duplicate_item_feeds,
            "duplicate_item_rows_excess": self.duplicate_item_rows_excess, "conflicting_target_pairs": self.conflicting_target_pairs,
            "feed_size": distribution(self.sizes), "history_size": distribution(self.history_sizes),
            "positives_per_feed": distribution(self.positives_per_feed),
            "dates_utc": {"min": datetime.fromtimestamp(self.min_timestamp, timezone.utc).isoformat(),
                          "max": datetime.fromtimestamp(self.max_timestamp, timezone.utc).isoformat()},
            "by_day_utc": dict(sorted(self.by_day.items())), "ctr_by_position": self.positions,
            "candidate_categories": dict(sorted(self.categories.items())),
            "history_event_types": dict(sorted(self.event_types.items())),
            "history_event_sources": dict(sorted(self.event_sources.items())), "features": moments,
        }


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_parquet_atomic(frame: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd", compression_level=3, statistics=True, row_group_size=100_000)
    temporary.replace(path)


def prepare_dataset(data_root: Path, output_root: Path, eda_root: Path, *, resume=False):
    data_root, output_root, eda_root = (Path(path).resolve() for path in (data_root, output_root, eda_root))
    if output_root == data_root or data_root in output_root.parents or output_root in data_root.parents:
        raise ValueError("prepared output and raw input must be separate directories")
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError("output is not empty; use --resume to reuse verified completed shards")
    started = time.monotonic()
    manifest = {"format_version": FORMAT_VERSION, "complete": False,
                "raw_root_relative": os.path.relpath(data_root, output_root),
                "candidate_key": ["split", "feed_id", "candidate_index"],
                "item_feature_join_key": ["split", "feed_id", "item_id"],
                "duplicate_item_policy": "preserve every original occurrence; candidate_index is metadata, not a model feature",
                "splits": {}}
    write_json(output_root / "manifest.json", manifest)
    statistics = {}
    for split in ("train", "val", "test"):
        state = SplitStatistics(split)
        parts = list_parts(data_root, split)
        meta_path = data_root / split / "item_embeddings/data.meta"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if len(meta["shape"]) != 2 or meta["shape"][1] != EMB_DIM or meta["dtype"] != "int8":
            raise ValueError(f"invalid embedding metadata: {split}")
        if (data_root / split / "item_embeddings/data.memmap").stat().st_size != math.prod(meta["shape"]):
            raise ValueError(f"invalid embedding file size: {split}")
        entries = []
        for index, source in enumerate(parts):
            frame = pl.read_parquet(source)
            validate_raw(frame, split, meta["shape"][0])
            state.update(frame)
            candidates = flatten_candidates(frame, split)
            duplicate_rows = candidates.filter(candidates.select('feed_id', 'item_id').is_duplicated())
            if duplicate_rows.height:
                expressions = [pl.len().alias('occurrences')]
                if split != 'test':
                    expressions.append(pl.col('target').n_unique().alias('target_values'))
                duplicate_groups = duplicate_rows.group_by('feed_id', 'item_id').agg(expressions)
                state.duplicate_item_pairs += duplicate_groups.height
                state.duplicate_item_feeds += duplicate_rows['feed_id'].n_unique()
                state.duplicate_item_rows_excess += duplicate_rows.height - duplicate_groups.height
                if split != 'test':
                    state.conflicting_target_pairs += duplicate_groups.filter(pl.col('target_values') > 1).height
                write_parquet_atomic(duplicate_rows, eda_root / 'duplicate_candidates' / split / source.name)
            expected_rows = int(frame["item_id"].list.len().sum())
            if candidates.height != expected_rows:
                raise ValueError(f"candidate row mismatch: {source}")
            relative_candidates = Path(split) / "candidates" / source.name
            relative_feeds = Path(split) / "feeds" / source.name
            candidate_path, feed_path = output_root / relative_candidates, output_root / relative_feeds
            checkpoint = output_root / "_state" / split / (source.stem + ".json")
            source_fingerprint = fingerprint(source)
            reuse = False
            if resume and checkpoint.exists() and candidate_path.exists() and feed_path.exists():
                previous = json.loads(checkpoint.read_text(encoding="utf-8"))
                reuse = (previous.get("source") == source_fingerprint and previous.get("format_version") == FORMAT_VERSION
                         and previous.get("candidate_file") == fingerprint(candidate_path)
                         and previous.get("feed_file") == fingerprint(feed_path)
                         and pq.read_metadata(candidate_path).num_rows == expected_rows
                         and pq.read_metadata(feed_path).num_rows == frame.height)
            if not reuse:
                write_parquet_atomic(candidates, candidate_path)
                feeds = frame.select(["feed_id", "timestamp", "user_id", "user_embedding"] + HISTORY_COLS).with_columns(
                    pl.lit(split).alias("split"), frame["item_id"].list.len().alias("feed_size"))
                write_parquet_atomic(feeds, feed_path)
                if pq.read_metadata(candidate_path).num_rows != expected_rows or pq.read_metadata(feed_path).num_rows != frame.height:
                    raise ValueError("written parquet row count mismatch")
                write_json(checkpoint, {"format_version": FORMAT_VERSION, "source": source_fingerprint,
                                       "candidate_file": fingerprint(candidate_path), "feed_file": fingerprint(feed_path)})
            entries.append({"source": source.name, "source_fingerprint": source_fingerprint, "feeds": frame.height,
                            "candidates": expected_rows, "candidate_path": relative_candidates.as_posix(),
                            "feed_path": relative_feeds.as_posix()})
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(parts):
                elapsed = time.monotonic() - started
                print(f"{split}: {index + 1}/{len(parts)} shards, {state.candidates:,} candidates, {elapsed:.0f}s", flush=True)
            del candidates, frame
        if state.feed_ids.count() != state.feeds:
            raise ValueError(f"duplicate or missing feed IDs in {split}")
        statistics[split] = state
        manifest["splits"][split] = {"parts": entries, "feeds": state.feeds, "candidates": state.candidates,
                                      "candidate_schema": {key: str(value) for key, value in pl.read_parquet_schema(candidate_path).items()},
                                      "feed_schema": {key: str(value) for key, value in pl.read_parquet_schema(feed_path).items()},
                                      "embedding_metadata": meta}
        write_json(output_root / "manifest.json", manifest)
        write_json(eda_root / f"{split}_summary.json", state.summary())
    # Exact distinct-ID overlap. Directional fractions always specify the denominator.
    overlaps = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        left, right = statistics[first], statistics[second]
        if left.feed_ids.intersection(right.feed_ids):
            raise ValueError(f"feed IDs overlap between {first} and {second}")
        for name, attribute in (("users", "users"), ("candidate_items", "items"), ("history_items", "history_items")):
            a, b = getattr(left, attribute), getattr(right, attribute)
            common = a.intersection(b)
            overlaps.append({"left": first, "right": second, "entity": name, "intersection": common,
                             "left_unique": a.count(), "right_unique": b.count(),
                             "fraction_of_left": common / a.count() if a.count() else 0.,
                             "fraction_of_right": common / b.count() if b.count() else 0.})
    # Check full test key coverage against the sample, not merely its row count.
    sample = pl.read_parquet(data_root / "sample_submission.parquet", columns=["feed_id", "item_id"])
    test_paths = [output_root / entry["candidate_path"] for entry in manifest["splits"]["test"]["parts"]]
    test_keys = pl.scan_parquet(test_paths).select("feed_id", "item_id").collect()
    if sample.height != test_keys.height or sample.is_duplicated().any() or sample.join(test_keys, on=["feed_id", "item_id"], how="anti").height:
        raise ValueError("prepared test candidate keys differ from sample_submission")
    manifest["complete"] = True
    manifest["test_template_key_coverage"] = "exact_match"
    manifest["duration_seconds"] = round(time.monotonic() - started, 2)
    summary = {"complete": True, "scope": "all shards, all feeds and all candidate/history list values; no sampling",
               "missing_id_policy": "ID 0 excluded from distinct counts and overlaps; occurrences reported separately",
               "date_timezone": "UTC", "splits": {name: value.summary() for name, value in statistics.items()},
               "candidate_key": manifest['candidate_key'], "duplicate_item_policy": manifest['duplicate_item_policy'],
               "overlaps": overlaps, "test_template_key_coverage": "exact_match",
               "prepared_manifest": os.path.relpath(output_root / "manifest.json", eda_root),
               "duration_seconds": manifest["duration_seconds"]}
    write_json(eda_root / "summary.json", summary)
    manifest["eda_summary_sha256"] = hashlib.sha256((eda_root / "summary.json").read_bytes()).hexdigest()
    write_json(output_root / "manifest.json", manifest)
    print(f"Complete: {output_root}; EDA: {eda_root / 'summary.json'}", flush=True)
    return summary


def scan_candidates(root: str | Path, split: str) -> pl.LazyFrame:
    """Read only prepared candidates lazily; no user/history vectors are repeated."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("prepared dataset is incomplete")
    parts = manifest["splits"][split]["parts"]
    return pl.scan_parquet([root / entry["candidate_path"] for entry in parts])
