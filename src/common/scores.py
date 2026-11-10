"""Работа со скорами: нормировка внутри ленты (скоры разных моделей несопоставимы по шкале)."""
from __future__ import annotations

import numpy as np
import polars as pl


def feed_zscore(feed_id: np.ndarray, score: np.ndarray) -> np.ndarray:
    d = pl.DataFrame({"f": feed_id, "s": np.asarray(score, dtype=np.float64)})
    out = d.with_columns(
        ((pl.col("s") - pl.col("s").mean().over("f")) / (pl.col("s").std().over("f") + 1e-9)).alias("z"))
    return out["z"].fill_null(0.0).to_numpy()


def feed_rank01(feed_id: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Ранг внутри ленты в (0, 1], 1 = лучший."""
    d = pl.DataFrame({"f": feed_id, "s": np.asarray(score, dtype=np.float64)})
    out = d.with_columns((pl.col("s").rank("average").over("f") / pl.len().over("f")).alias("r"))
    return out["r"].to_numpy()
