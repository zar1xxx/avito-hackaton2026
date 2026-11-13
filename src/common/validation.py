"""Feed-level time holdout and reproducible OOF folds for val."""
from __future__ import annotations

import math

import numpy as np
import polars as pl


def make_validation_split(
    frame: pl.DataFrame, public_fraction: float = 0.3, n_folds: int = 5, seed: int = 42,
) -> pl.DataFrame:
    """Return one assignment per feed; equal timestamps stay on one holdout side.

    Folds group by feed_id, not user_id, and are not chronological. Features
    derived from targets must still be computed using only past data or OOF.
    """
    if not math.isfinite(public_fraction) or not 0 < public_fraction < 1:
        raise ValueError("public_fraction must be between 0 and 1")
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    feeds = frame.select("feed_id", "timestamp")
    if feeds.null_count().to_numpy().any():
        raise ValueError("feed_id and timestamp must not contain nulls")
    if not feeds.schema["feed_id"].is_integer():
        raise ValueError("feed_id must be an integer")
    time_type = feeds.schema["timestamp"]
    if not (time_type.is_numeric() or time_type.is_temporal()):
        raise ValueError("timestamp must be numeric or temporal, not a string")
    if time_type.is_float() and not feeds["timestamp"].is_finite().all():
        raise ValueError("timestamp must be finite")
    feeds = feeds.unique().sort("timestamp", "feed_id")
    if feeds["feed_id"].n_unique() != feeds.height:
        raise ValueError("a feed_id has conflicting timestamps")
    if feeds.height < n_folds:
        raise ValueError("not enough feeds for n_folds")
    times = feeds["timestamp"].to_list()
    boundaries = [i for i in range(1, len(times)) if times[i - 1] != times[i]]
    if not boundaries:
        raise ValueError("at least two distinct timestamps are needed for a time holdout")
    cut = min(boundaries, key=lambda i: (abs(i - feeds.height * public_fraction), i))
    # Assign folds in sorted feed_id order so input ordering cannot affect OOF.
    ids = feeds["feed_id"].sort()
    rng = np.random.default_rng(seed)
    fold = np.empty(feeds.height, dtype=np.int32)
    fold[rng.permutation(feeds.height)] = np.arange(feeds.height) % n_folds
    assignments = pl.DataFrame({"feed_id": ids, "fold": fold})
    return (
        feeds.with_columns(pl.Series("leaderboard_split", ["public"] * cut + ["private"] * (feeds.height - cut)))
        .join(assignments, on="feed_id", how="left")
        .sort("timestamp", "feed_id")
    )
