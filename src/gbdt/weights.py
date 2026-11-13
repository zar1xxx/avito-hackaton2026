"""Веса против смещения позиции в train (IPW).

В train кандидаты стоят в порядке боевого рекомендатора, поэтому верхние позиции получают клики
отчасти просто за счёт места. В val/test порядок случайный. Грубая оценка вероятности «увидеть»
позицию k: p_k = CTR_k / CTR_ref (ref = средний CTR позиций 0-2), обрезана в [1/max_w, 1].
Позитивам даём вес 1/p_k — клик на низкой позиции «дороже».

ВАЖНО: это приближение. CTR по позиции смешивает «видимость» и «релевантность»
(наверху стоит более релевантное), поэтому метод надо проверять на val, а не принимать на веру.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def position_ctr(pos: np.ndarray, target: np.ndarray, smooth: int = 5, min_count: int = 30) -> np.ndarray:
    # pos с пропусками приходит из numpy как float с NaN: убираем NaN/null и возвращаем целые
    d = (pl.DataFrame({"pos": np.asarray(pos, dtype=np.float64), "t": target})
         .filter(pl.col("pos").is_not_null() & pl.col("pos").is_not_nan() & (pl.col("pos") >= 0))
         .with_columns(pl.col("pos").cast(pl.Int64)))
    g = d.group_by("pos").agg(pl.col("t").mean().alias("ctr"), pl.len().alias("n")).sort("pos")
    maxp = int(g["pos"].max())
    ctr = np.full(maxp + 1, np.nan)
    ctr[g["pos"].to_numpy()] = np.where(g["n"].to_numpy() >= min_count, g["ctr"].to_numpy(), np.nan)
    last = np.nan
    for i in range(len(ctr)):  # протягиваем последнее надёжное значение вперёд
        if np.isnan(ctr[i]):
            ctr[i] = last
        else:
            last = ctr[i]
    ctr = np.where(np.isnan(ctr), np.nanmean(ctr), ctr)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pad = np.pad(ctr, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
        ctr = np.convolve(pad, k, mode="valid")
    return ctr


def ipw_weights(pos: np.ndarray, target: np.ndarray, ctr: np.ndarray, max_w: float = 10.0,
                positives_only: bool = True) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    ref = ctr[: min(3, len(ctr))].mean()
    prop = np.clip(ctr / max(ref, 1e-9), 1.0 / max_w, 1.0)
    p = np.ones(len(pos))
    ok = ~np.isnan(pos) & (pos >= 0)
    idx = np.minimum(pos[ok].astype(np.int64), len(ctr) - 1)
    p[ok] = prop[idx]
    w = 1.0 / p
    if positives_only:
        w = np.where(np.asarray(target) == 1, w, 1.0)
    return w.astype(np.float32)
