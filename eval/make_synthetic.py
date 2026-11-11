"""Генератор синтетического датасета в формате соревнования — только чтобы гонять и отлаживать код.

Это НЕ данные Авито и НЕ модель их поведения. Что повторяет формат и свойства из описания:
- train: кандидаты стоят в порядке «боевого рекомендатора» (есть pos, часть pos пропущена),
  вероятность клика падает с позицией (смещение показа);
- val/test: порядок случайный, смещения нет, кандидаты перемешаны; pos нет;
- у каждого сплита своя таблица эмбеддингов с другими индексами, item_id общий; часть эмбеддингов = -1;
- пользователи и объявления пересекаются между сплитами, ленты идут по времени, несколько parquet-частей.

    python eval/make_synthetic.py --out data/synth --feeds 20000 --parts 4
Числа качества на этих данных ничего не говорят о качестве на реальных.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

D = 128
N_ITEMS = 5000
N_USERS = 800
N_CATS = 30


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_split(root: Path, name: str, n_feeds: int, parts: int, rng: np.random.Generator, glob_emb: np.ndarray,
               item_cat: np.ndarray, users: np.ndarray, feed0: int, t0: int, biased: bool, with_target: bool,
               ) -> None:
    d = root / name
    (d / "impressions").mkdir(parents=True, exist_ok=True)
    (d / "item_embeddings").mkdir(parents=True, exist_ok=True)

    perm = rng.permutation(N_ITEMS)             # строка таблицы -> item
    inv = np.empty(N_ITEMS, dtype=np.int64)     # item -> строка таблицы
    inv[perm] = np.arange(N_ITEMS)
    glob_emb[perm].astype(np.int8).tofile(d / "item_embeddings" / "data.memmap")
    (d / "item_embeddings" / "data.meta").write_text(
        json.dumps({"dtype": "int8", "shape": [N_ITEMS, D], "order": "C"}))
    no_emb = rng.random(N_ITEMS) < 0.03         # у части объявлений нет эмбеддинга

    rows = []
    for i in range(n_feeds):
        m = int(rng.integers(5, 60))
        uid = int(rng.integers(1, N_USERS + 1))
        user = users[uid - 1]
        items = rng.choice(N_ITEMS, size=m, replace=False)
        cand = glob_emb[items]
        cos = cand @ user / (np.linalg.norm(cand, axis=1) * np.linalg.norm(user) + 1e-6)
        f = rng.normal(size=(m, 5)).astype(np.float32)
        f[:, 0] += 0.5 * cos                    # слабый сигнал в f1
        logit = 5 * cos + 0.4 * f[:, 0] + rng.normal(scale=0.7, size=m) - 3.5

        if biased:  # порядок боевого рекомендатора: по «своему» скору с шумом, клики затухают по позиции
            order = np.argsort(-(logit + rng.normal(scale=1.0, size=m)))
            items, f, logit = items[order], f[order], logit[order]
            pos = np.arange(m)
            exam = 1.0 / (1.0 + 0.25 * pos)
            target = (rng.random(m) < _sigmoid(logit) * exam).astype(np.int8)
        else:       # случайный порядок, кандидаты уже перемешаны
            sh = rng.permutation(m)
            items, f, logit = items[sh], f[sh], logit[sh]
            pos = None
            target = (rng.random(m) < _sigmoid(logit)).astype(np.int8)
        if with_target and target.sum() == 0:
            if not biased:
                target[rng.integers(m)] = 1
            else:
                target[0] = 1
        h = int(rng.integers(0, 80))
        h_items = rng.choice(N_ITEMS, size=h) if h else np.array([], dtype=np.int64)
        # история чаще содержит близкие к пользователю товары и иногда сам кандидат
        if h and rng.random() < 0.3:
            h_items[: min(h, 3)] = items[np.argsort(-logit)[: min(h, 3)]]
        cidx = np.where(no_emb[items], -1, inv[items])
        hidx = np.where(no_emb[h_items], -1, inv[h_items]) if h else np.array([], dtype=np.int64)
        row = {
            "feed_id": feed0 + i,
            "timestamp": t0 + i * 20,
            "user_id": uid,
            "category_id": item_cat[items].astype(np.int8).tolist(),
            **{f"f{j + 1}": f[:, j].tolist() for j in range(5)},
            "item_id__embedding_idx": cidx.astype(np.int32).tolist(),
            "item_id": (items + 1).astype(np.int32).tolist(),
            "user_embedding": user.astype(np.int8).tolist(),
            "history__item_id": (h_items + 1).astype(np.int32).tolist(),
            "history__category_id": (item_cat[h_items] if h else np.array([], dtype=np.int64)).astype(np.int8).tolist(),
            "history__event_type": rng.integers(1, 5, size=h).astype(np.int8).tolist(),
            "history__event_source": rng.integers(1, 4, size=h).astype(np.int8).tolist(),
            "history__item_id__embedding_idx": hidx.astype(np.int32).tolist(),
        }
        if with_target:
            row["target"] = target.tolist()
        if biased:
            p = pos.astype(np.int16).tolist()
            for k in np.flatnonzero(rng.random(m) < 0.05):   # пропуски в pos после фильтрации
                p[k] = None
            row["pos"] = p
        rows.append(row)

    schema_over = {"pos": pl.List(pl.Int16)} if biased else {}
    chunks = np.array_split(np.arange(len(rows)), parts)
    for k, ch in enumerate(chunks):
        pl.DataFrame([rows[j] for j in ch], schema_overrides=schema_over).write_parquet(
            d / "impressions" / f"part-{k:05d}.parquet", compression="zstd")
    (d / "impressions" / "_SUCCESS").touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feeds", type=int, default=3000, help="лент в train; val и test = feeds/4")
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    glob_emb = rng.integers(-100, 100, size=(N_ITEMS, D)).astype(np.float32)
    users = rng.integers(-100, 100, size=(N_USERS, D)).astype(np.float32)
    item_cat = rng.integers(1, N_CATS, size=N_ITEMS)
    root = Path(a.out)
    nv = max(a.feeds // 4, 10)
    make_split(root, "train", a.feeds, a.parts, rng, glob_emb, item_cat, users, 0, 1_788_000_000, True, True)
    make_split(root, "val", nv, max(a.parts // 2, 1), rng, glob_emb, item_cat, users, 10**6, 1_790_000_000, False, True)
    make_split(root, "test", nv, max(a.parts // 2, 1), rng, glob_emb, item_cat, users, 2 * 10**6, 1_791_000_000,
               False, False)
    t = pl.concat([pl.read_parquet(p, columns=["feed_id", "item_id"])
                   for p in sorted((root / "test" / "impressions").glob("part-*.parquet"))]
                  ).explode("item_id")
    t = t.with_columns(pl.lit(0.0).cast(pl.Float32).alias("score")).cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    t.write_parquet(root / "sample_submission.parquet")
    print("готово:", root, "| train лент:", a.feeds, "| val/test лент:", nv)


if __name__ == "__main__":
    main()
