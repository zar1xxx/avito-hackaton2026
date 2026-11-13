"""NDCG по правилам соревнования Авито.

Правила:
- внутри ленты сортируем по убыванию score; при равных score негативы раньше позитивов;
- лента без позитивов -> NDCG = 1;
- итог = среднее по лентам с равным весом.

Эталон competition/baseline/calculate_metrics.py считает full, @10 и @30.
При cutoff=k и DCG, и IDCG ограничены первыми k кандидатами:
- idcg_k=None   : IDCG по всем позитивам (классический NDCG без отсечения)
- idcg_k=30     : экспериментальный вариант из описания, не метрика эталона
- cutoff=30     : и DCG, и IDCG только по первым 30 позициям (NDCG@30)
"""
from __future__ import annotations

import numpy as np


def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))


def _validate(target: np.ndarray, score: np.ndarray, cutoff: int | None, idcg_k: int | None) -> None:
    if target.ndim != 1 or score.ndim != 1 or target.shape != score.shape:
        raise ValueError("target and score must be one-dimensional arrays of equal length")
    if not len(target):
        raise ValueError("cannot evaluate an empty set of candidates")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("target must contain only binary labels (0/1)")
    if not np.isfinite(score).all():
        raise ValueError("score must be finite")
    for name, value in (("cutoff", cutoff), ("idcg_k", idcg_k)):
        if value is not None and (not isinstance(value, (int, np.integer)) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if cutoff is not None and idcg_k is not None:
        raise ValueError("use either cutoff or idcg_k, not both")


def ndcg_one(
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    _validate(target, score, cutoff, idcg_k)
    n = len(target)
    if target.sum() == 0:
        return 1.0
    # тай-брейк: при равных score негатив выше позитива.
    # lexsort сортирует по последнему ключу первым: сначала -score, затем target (0 раньше 1)
    order = np.lexsort((target, -score))
    disc = _discounts(n)
    ranked = target[order]
    if cutoff is not None:
        dcg = float((ranked[:cutoff] * disc[:cutoff]).sum())
    else:
        dcg = float((ranked * disc).sum())
    ideal = np.sort(target)[::-1]
    k = cutoff if cutoff is not None else idcg_k
    if k is not None:
        idcg = float((ideal[:k] * disc[:k]).sum())
    else:
        idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else 1.0


def per_feed_ndcg(
    feed_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NDCG каждой ленты: (отсортированные feed_id, значения). Нужно для бутстрепа и сравнения моделей."""
    feed_ids = np.asarray(feed_ids)
    target = np.asarray(target)
    score = np.asarray(score)
    order = np.argsort(feed_ids, kind="stable")
    f = feed_ids[order]
    bounds = np.flatnonzero(np.diff(f)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(f)]])
    t, s = target[order], score[order]
    vals = np.array([ndcg_one(t[a:b], s[a:b], cutoff, idcg_k) for a, b in zip(starts, ends)])
    return f[starts], vals


def mean_ndcg(
    feed_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    """Среднее NDCG по лентам. Массивы выровнены по строкам."""
    feed_ids = np.asarray(feed_ids)
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    _validate(target, score, cutoff, idcg_k)
    if feed_ids.ndim != 1 or len(feed_ids) != len(target):
        raise ValueError("feed_ids must be a one-dimensional array aligned with target and score")
    if not np.issubdtype(feed_ids.dtype, np.integer):
        raise ValueError("feed_ids must contain integer IDs")
    order = np.argsort(feed_ids, kind="stable")
    f = feed_ids[order]
    bounds = np.flatnonzero(np.diff(f)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(f)]])
    t, s = target[order], score[order]
    vals = [ndcg_one(t[a:b], s[a:b], cutoff, idcg_k) for a, b in zip(starts, ends)]
    return float(np.mean(vals))


def all_variants(feed_ids, target, score) -> dict[str, float]:
    return {
        "ndcg_full": mean_ndcg(feed_ids, target, score),
        "ndcg_idcg30": mean_ndcg(feed_ids, target, score, idcg_k=30),
        "ndcg@10": mean_ndcg(feed_ids, target, score, cutoff=10),
        "ndcg@30": mean_ndcg(feed_ids, target, score, cutoff=30),
    }
