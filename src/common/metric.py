"""NDCG по правилам соревнования Авито.

Правила:
- внутри ленты сортируем по убыванию score; при равных score негативы раньше позитивов;
- лента без позитивов -> NDCG = 1;
- итог = среднее по лентам с равным весом.

В описании сказано: DCG считается по всем кандидатам, а IDCG@30 — по первым 30.
Это нестыковка, поэтому поддерживаем оба режима и сверяем с calculate_metrics.py:
- idcg_k=None   : IDCG по всем позитивам (классический NDCG без отсечения)
- idcg_k=30     : IDCG@30, DCG по всем кандидатам (буквально как в тексте)
- cutoff=30     : и DCG, и IDCG только по первым 30 позициям (NDCG@30)
"""
from __future__ import annotations

import numpy as np


def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, n + 2))


def ndcg_one(
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
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


def mean_ndcg(
    feed_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    cutoff: int | None = None,
    idcg_k: int | None = None,
) -> float:
    """Среднее NDCG по лентам. Массивы выровнены по строкам."""
    feed_ids = np.asarray(feed_ids)
    target = np.asarray(target)
    score = np.asarray(score)
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
        "ndcg@30": mean_ndcg(feed_ids, target, score, cutoff=30),
    }
