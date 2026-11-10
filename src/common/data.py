"""Чтение лент и построение табличных признаков (одна строка = кандидат в ленте).

Ленты читаются по одному parquet-файлу (part-*.parquet), чтобы не держать 19 ГБ в памяти.
Эмбеддинги — int8 memmap, индекс -1 = нет эмбеддинга (в float32 переводим до любых операций).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

EMB_DIM = 128
HIST_RECENT = 10
CAND_COLS = ["category_id", "f1", "f2", "f3", "f4", "f5", "item_id", "item_id__embedding_idx"]
FLOAT_FEATS = ["f1", "f2", "f3", "f4", "f5"]


def list_parts(data_root: str | Path, split: str) -> list[Path]:
    parts = sorted((Path(data_root) / split / "impressions").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"нет part-*.parquet в {data_root}/{split}/impressions")
    return parts


def load_embeddings(data_root: str | Path, split: str) -> np.memmap:
    d = Path(data_root) / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(
        d / "data.memmap", dtype=meta["dtype"], mode="r",
        shape=tuple(meta["shape"]), order=meta["order"],
    )


def _gather(emb: np.memmap, idx: np.ndarray) -> np.ndarray:
    """Эмбеддинги по индексам; для -1 нули. Всегда float32."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.zeros((len(idx), EMB_DIM), dtype=np.float32)
    ok = idx >= 0
    if ok.any():
        out[ok] = emb[idx[ok]].astype(np.float32)
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-6)


def _feed_features(row: dict, emb: np.memmap) -> dict[str, np.ndarray]:
    """Признаки сходства и пересечения с историей для всех кандидатов одной ленты."""
    cand_idx = np.asarray(row["item_id__embedding_idx"], dtype=np.int64)
    cand_item = np.asarray(row["item_id"], dtype=np.int64)
    cand_cat = np.asarray(row["category_id"], dtype=np.int64)
    m = len(cand_idx)
    ce = _gather(emb, cand_idx)
    has_emb = (cand_idx >= 0).astype(np.float32)
    ue = np.asarray(row["user_embedding"], dtype=np.float32)

    cos_u = _unit(ce) @ _unit(ue[None, :])[0]
    dot_u = ce @ ue
    out = {
        "has_emb": has_emb,
        "cos_user": cos_u * has_emb,
        "dot_user": dot_u,
        "item_norm": np.linalg.norm(ce, axis=1),
    }

    h_item = np.asarray(row["history__item_id"] or [], dtype=np.int64)
    h_cat = np.asarray(row["history__category_id"] or [], dtype=np.int64)
    h_idx = np.asarray(row["history__item_id__embedding_idx"] or [], dtype=np.int64)
    h = len(h_item)
    out["hist_len"] = np.full(m, h, dtype=np.float32)

    if h == 0:
        for k in ("cos_hist_mean", "cos_hist_max", "cos_hist_recent_max", "cos_hist_recent_mean",
                  "item_in_hist", "item_hist_first_pos", "cat_share_hist", "cat_in_recent"):
            out[k] = np.zeros(m, dtype=np.float32)
        out["item_hist_first_pos"][:] = -1.0
        return out

    he = _gather(emb, h_idx)
    h_ok = h_idx >= 0
    cu = _unit(ce)
    if h_ok.any():
        hu = _unit(he[h_ok])
        sim = cu @ hu.T  # (M, H_ok)
        out["cos_hist_mean"] = (cu @ hu.mean(0)) * has_emb
        out["cos_hist_max"] = sim.max(1) * has_emb
        rec = hu[:HIST_RECENT]
        s_rec = cu @ rec.T
        out["cos_hist_recent_max"] = s_rec.max(1) * has_emb
        out["cos_hist_recent_mean"] = s_rec.mean(1) * has_emb
    else:
        for k in ("cos_hist_mean", "cos_hist_max", "cos_hist_recent_max", "cos_hist_recent_mean"):
            out[k] = np.zeros(m, dtype=np.float32)

    # пересечение кандидата с историей (история идёт от новых к старым)
    eq = cand_item[:, None] == h_item[None, :]
    out["item_in_hist"] = eq.sum(1).astype(np.float32)
    first = np.where(eq.any(1), eq.argmax(1), -1)
    out["item_hist_first_pos"] = first.astype(np.float32)

    # категории
    cat_eq = cand_cat[:, None] == h_cat[None, :]
    out["cat_share_hist"] = cat_eq.mean(1).astype(np.float32)
    out["cat_in_recent"] = cat_eq[:, :HIST_RECENT].mean(1).astype(np.float32)
    return out


def _in_feed_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Ранги и z-score признаков внутри ленты + размер ленты."""
    exprs = [pl.len().over("feed_id").alias("feed_size")]
    for c in FLOAT_FEATS + ["cos_user", "cos_hist_max"]:
        exprs.append((pl.col(c).rank("average").over("feed_id") / pl.len().over("feed_id")).alias(f"{c}_rank"))
        exprs.append(
            ((pl.col(c) - pl.col(c).mean().over("feed_id")) / (pl.col(c).std().over("feed_id") + 1e-6))
            .alias(f"{c}_z")
        )
    exprs.append((pl.col("category_id").count().over("feed_id", "category_id")
                  / pl.len().over("feed_id")).alias("cat_share_feed"))
    return df.with_columns(exprs)


def build_part(path: Path, emb: np.memmap) -> pl.DataFrame:
    """Одна часть -> таблица кандидатов с признаками. Порядок строк = порядок лент и кандидатов."""
    feeds = pl.read_parquet(path)
    has_target = "target" in feeds.columns
    has_pos = "pos" in feeds.columns

    cols = ["feed_id", "timestamp", "user_id"] + CAND_COLS
    if has_target:
        cols.append("target")
    if has_pos:
        cols.append("pos")
    long = feeds.select(cols).explode(CAND_COLS + (["target"] if has_target else []) + (["pos"] if has_pos else []))

    keys = ["item_id__embedding_idx", "item_id", "category_id", "user_embedding",
            "history__item_id", "history__category_id", "history__item_id__embedding_idx"]
    per_feed: list[dict[str, np.ndarray]] = []
    for row in feeds.select(keys).iter_rows(named=True):
        per_feed.append(_feed_features(row, emb))
    names = list(per_feed[0].keys())
    feat = {k: np.concatenate([d[k] for d in per_feed]).astype(np.float32) for k in names}
    long = long.with_columns([pl.Series(k, v) for k, v in feat.items()])
    long = _in_feed_stats(long)
    return long


def iter_split(data_root: str | Path, split: str, max_parts: int | None = None) -> Iterator[pl.DataFrame]:
    emb = load_embeddings(data_root, split)
    for p in list_parts(data_root, split)[:max_parts]:
        yield build_part(p, emb)


def load_split(data_root: str | Path, split: str, max_parts: int | None = None) -> pl.DataFrame:
    return pl.concat(list(iter_split(data_root, split, max_parts)), how="vertical")


MODEL_FEATURES = (
    FLOAT_FEATS
    + ["category_id", "has_emb", "cos_user", "dot_user", "item_norm", "hist_len",
       "cos_hist_mean", "cos_hist_max", "cos_hist_recent_max", "cos_hist_recent_mean",
       "item_in_hist", "item_hist_first_pos", "cat_share_hist", "cat_in_recent", "feed_size",
       "cat_share_feed"]
    + [f"{c}_{s}" for c in FLOAT_FEATS + ["cos_user", "cos_hist_max"] for s in ("rank", "z")]
)
