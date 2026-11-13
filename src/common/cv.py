"""Разбиения лент по времени: псевдо-Public/Private и блочные временные фолды.

Public/Private в соревновании делятся по времени 30/70, поэтому на val делаем так же:
подбор (early stopping, веса ансамбля) — на «псевдо-Public», оценка — на «псевдо-Private».
Ленты с одинаковым timestamp остаются в одной части.
"""
from __future__ import annotations

import numpy as np


def feed_times(feed_id: np.ndarray, timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uf, idx = np.unique(np.asarray(feed_id), return_index=True)
    return uf, np.asarray(timestamp)[idx]


def pseudo_split(feed_id, timestamp, frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Ранние ~frac лент -> pseudo-public, остальные -> pseudo-private (списки feed_id)."""
    uf, ut = feed_times(feed_id, timestamp)
    order = np.argsort(ut, kind="stable")
    n_pub = int(round(len(uf) * frac))
    if n_pub == 0:
        return uf[:0], uf
    cut = ut[order][n_pub - 1]
    return uf[ut <= cut], uf[ut > cut]


def time_folds(feed_id, timestamp, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k последовательных по времени блоков равного числа лент. Возвращает (feed_id, fold)."""
    uf, ut = feed_times(feed_id, timestamp)
    order = np.argsort(ut, kind="stable")
    fold = np.empty(len(uf), dtype=np.int64)
    fold[order] = np.minimum(np.arange(len(uf)) * k // len(uf), k - 1)
    return uf, fold


def mask_feeds(feed_id: np.ndarray, subset: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(feed_id), subset)
