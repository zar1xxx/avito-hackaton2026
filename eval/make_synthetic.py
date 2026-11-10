"""Генератор крошечного датасета в формате соревнования — только чтобы проверить код.

Данные Авито здесь НЕ используются: сигнал синтетический (косинус user·item + шум).
    python eval/make_synthetic.py --out /tmp/synth --feeds 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

D = 128


def make_split(root: Path, name: str, n_feeds: int, rng: np.random.Generator, feed0: int,
               with_pos: bool, with_target: bool) -> None:
    n_items = 2000
    emb = rng.integers(-100, 100, size=(n_items, D)).astype(np.int8)
    d = root / name
    (d / "impressions").mkdir(parents=True, exist_ok=True)
    (d / "item_embeddings").mkdir(parents=True, exist_ok=True)
    emb.tofile(d / "item_embeddings" / "data.memmap")
    (d / "item_embeddings" / "data.meta").write_text(
        json.dumps({"dtype": "int8", "shape": [n_items, D], "order": "C"}))

    rows = []
    for i in range(n_feeds):
        m = int(rng.integers(5, 40))
        user = rng.integers(-100, 100, size=D).astype(np.int8)
        idx = rng.choice(n_items, size=m, replace=False)
        cand = emb[idx].astype(np.float32)
        cos = cand @ user.astype(np.float32) / (np.linalg.norm(cand, axis=1) * np.linalg.norm(user) + 1e-6)
        logit = 6 * cos + rng.normal(size=m) - 3
        target = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(np.int8)
        if with_target and target.sum() == 0:
            target[rng.integers(m)] = 1
        h = int(rng.integers(0, 60))
        h_idx = rng.choice(n_items, size=h) if h else np.array([], dtype=np.int64)
        row = {
            "feed_id": feed0 + i,
            "timestamp": 1_790_000_000 + i * 60,
            "user_id": int(rng.integers(1, 200)),
            "category_id": rng.integers(1, 20, size=m).astype(np.int8).tolist(),
            **{f"f{j}": rng.normal(size=m).astype(np.float32).tolist() for j in range(1, 6)},
            "item_id__embedding_idx": idx.astype(np.int32).tolist(),
            "item_id": (idx + 1).astype(np.int32).tolist(),
            "user_embedding": user.tolist(),
            "history__item_id": (h_idx + 1).astype(np.int32).tolist(),
            "history__category_id": rng.integers(1, 20, size=h).astype(np.int8).tolist(),
            "history__event_type": rng.integers(1, 5, size=h).astype(np.int8).tolist(),
            "history__event_source": rng.integers(1, 4, size=h).astype(np.int8).tolist(),
            "history__item_id__embedding_idx": h_idx.astype(np.int32).tolist(),
        }
        if with_target:
            row["target"] = target.tolist()
        if with_pos:
            row["pos"] = rng.permutation(m).astype(np.int16).tolist()
        rows.append(row)
    pl.DataFrame(rows).write_parquet(d / "impressions" / "part-00000.parquet", compression="zstd")
    (d / "impressions" / "_SUCCESS").touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feeds", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    root = Path(a.out)
    make_split(root, "train", a.feeds, rng, 0, with_pos=True, with_target=True)
    make_split(root, "val", a.feeds // 3, rng, 10**6, with_pos=False, with_target=True)
    make_split(root, "test", a.feeds // 3, rng, 2 * 10**6, with_pos=False, with_target=False)
    t = pl.read_parquet(root / "test" / "impressions" / "part-00000.parquet").select(
        "feed_id", "item_id").explode("item_id").with_columns(pl.lit(0.0).cast(pl.Float32).alias("score"))
    t.cast({"feed_id": pl.Int32, "item_id": pl.Int32}).write_parquet(root / "sample_submission.parquet")
    print("готово:", root)


if __name__ == "__main__":
    main()
