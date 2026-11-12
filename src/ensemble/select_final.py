"""Выбор двух финальных решений по val.

Метрика — NDCG@30 (как в эталоне calculate_metrics.py).
Принцип: ориентируемся на pseudo-private, а не на Public лидерборда (Public — всего ~30% лент).
Для каждого кандидата считаем NDCG, парный бутстреп против лидера (по лентам pseudo-private) и
корреляцию с лидером. Финал A — лучший по pseudo-private. Финал B — лучший «безопасный» кандидат
(--safe: решения без рискованных признаков, напр. без истории «из будущего»), а если безопасных нет —
самый непохожий на A среди тех, кто не хуже A статистически.

  PYTHONPATH=. python -m src.ensemble.select_final --data competition \
      --runs runs/cb runs/lgb runs/blend --safe runs/cb runs/lgb
Кандидаты — папки с val_pred.parquet, содержащей OOF-предсказания на val (колонки feed_id,item_id,score,target).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import pseudo_split
from src.common.data import load_labels
from src.common.metric import per_feed_ndcg
from src.common.scores import feed_zscore

KEYS = ["feed_id", "item_id"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--safe", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--public-lb", nargs="*", default=[], help="name=значение Public NDCG, напр. cb=0.412")
    a = ap.parse_args(argv)

    lab = load_labels(a.data, "val").cast({"feed_id": pl.Int32, "item_id": pl.Int32}).sort("feed_id", maintain_order=True)
    pub, priv = pseudo_split(lab["feed_id"].to_numpy(), lab["timestamp"].to_numpy())
    f = lab["feed_id"].to_numpy()
    mpriv = np.isin(f, priv)
    lb = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.public_lb}

    rows, perfeed, zs = {}, {}, {}
    for r in a.runs:
        p = pl.read_parquet(Path(r) / "val_pred.parquet").select(KEYS + ["score"]).cast(
            {"feed_id": pl.Int32, "item_id": pl.Int32})
        df = lab.join(p, on=KEYS, how="left")
        if df["score"].null_count():
            raise ValueError(f"{r}: нет предсказаний для части val")
        s = df["score"].to_numpy()
        fid, vals = per_feed_ndcg(f[mpriv], df["target"].to_numpy()[mpriv], s[mpriv], cutoff=30)
        perfeed[Path(r).name] = vals
        zs[Path(r).name] = feed_zscore(f, s)
        _, vp = per_feed_ndcg(f[~mpriv], df["target"].to_numpy()[~mpriv], s[~mpriv], cutoff=30)
        rows[Path(r).name] = {"pub": float(vp.mean()), "priv": float(vals.mean()),
                              "priv_se": float(vals.std(ddof=1) / np.sqrt(len(vals)))}
    names = list(rows)
    lead = max(names, key=lambda n: rows[n]["priv"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(perfeed[lead]), size=(a.boot, len(perfeed[lead])))
    for n in names:
        diff = (perfeed[n][idx] - perfeed[lead][idx]).mean(1)
        rows[n]["p_beats_lead"] = float((diff > 0).mean()) if n != lead else float("nan")
        rows[n]["corr_with_lead"] = float(np.corrcoef(zs[n], zs[lead])[0, 1])

    safe = [Path(s).name for s in a.safe]
    cand_b = [n for n in safe if n != lead]
    if cand_b:
        b = max(cand_b, key=lambda n: rows[n]["priv"])
        why = "лучший безопасный кандидат"
    else:
        near = [n for n in names if n != lead and rows[n]["p_beats_lead"] > 0.05]
        b = min(near, key=lambda n: rows[n]["corr_with_lead"]) if near else None
        why = "самый непохожий на A среди статистически не худших"

    print(f"{'модель':14s} {'pub':>7s} {'priv':>7s} {'±se':>7s} {'P(>лидер)':>10s} {'corr':>6s} {'Public LB':>10s}")
    for n in sorted(names, key=lambda n: -rows[n]["priv"]):
        r = rows[n]
        print(f"{n:14s} {r['pub']:7.4f} {r['priv']:7.4f} {r['priv_se']:7.4f} {r['p_beats_lead']:10.2f} "
              f"{r['corr_with_lead']:6.2f} {lb.get(n, float('nan')):10.4f}")
    print(f"\nФинал A: {lead} (лучший по pseudo-private)")
    print(f"Финал B: {b} ({why})" if b else "Финал B: подходящего кандидата нет — нужен ещё один непохожий на A")
    if lb and len(lb) >= 3:
        xs = np.array([rows[n]["priv"] for n in lb if n in rows])
        ys = np.array([lb[n] for n in lb if n in rows])
        print(f"корреляция pseudo-private и Public LB по {len(xs)} моделям: {np.corrcoef(xs, ys)[0, 1]:.2f}"
              " (низкая = val плохо предсказывает Public, ищем причину)")


if __name__ == "__main__":
    main()
