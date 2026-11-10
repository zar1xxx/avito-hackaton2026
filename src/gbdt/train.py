"""CatBoost-ранкер (YetiRank) на табличных признаках.

    python -m src.gbdt.train train   --data competition --out runs/gbdt-01 [--pos-mode none|feature]
    python -m src.gbdt.train predict --data competition --model runs/gbdt-01/model.cbm --out runs/gbdt-01/submission.parquet

pos-mode:
  none     — pos не используется (train смещён порядком показа, на test его нет)
  feature  — pos как признак при обучении; на инференсе подставляется константа --pos-const
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from src.common.data import MODEL_FEATURES, load_split
from src.common.metric import all_variants

CAT = ["category_id"]


def _prep(df: pl.DataFrame, pos_mode: str, pos_const: float) -> tuple[pl.DataFrame, list[str]]:
    feats = list(MODEL_FEATURES)
    if pos_mode == "feature":
        if "pos" in df.columns:
            df = df.with_columns(pl.col("pos").fill_null(-1).cast(pl.Float32))
        else:
            df = df.with_columns(pl.lit(pos_const).cast(pl.Float32).alias("pos"))
        feats.append("pos")
    df = df.with_columns(pl.col("category_id").cast(pl.Int32))
    return df.sort("feed_id", maintain_order=True), feats


def _pool(df: pl.DataFrame, feats: list[str], with_label: bool) -> Pool:
    x = df.select(feats).to_pandas()
    return Pool(x, label=df["target"].to_numpy() if with_label else None,
                group_id=df["feed_id"].to_numpy(), cat_features=CAT)


def train(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, feats = _prep(load_split(a.data, "train", a.max_parts), a.pos_mode, a.pos_const)
    va, _ = _prep(load_split(a.data, "val", a.max_parts), a.pos_mode, a.pos_const)
    params = dict(
        loss_function=a.loss, iterations=a.iterations, learning_rate=a.lr, depth=a.depth,
        eval_metric="NDCG", random_seed=a.seed, task_type=a.task_type,
        od_type="Iter", od_wait=100, verbose=100,
    )
    model = CatBoostRanker(**params)
    model.fit(_pool(tr, feats, True), eval_set=_pool(va, feats, True), use_best_model=True)
    model.save_model(str(out / "model.cbm"))

    score = model.predict(_pool(va, feats, False))
    res = all_variants(va["feed_id"].to_numpy(), va["target"].to_numpy(), score)
    res.update(vars(a) | {"n_train_rows": tr.height, "n_val_rows": va.height})
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print({k: v for k, v in res.items() if k.startswith("ndcg")})


def predict(a: argparse.Namespace) -> None:
    te, feats = _prep(load_split(a.data, "test", a.max_parts), a.pos_mode, a.pos_const)
    model = CatBoostRanker()
    model.load_model(a.model)
    score = model.predict(_pool(te, feats, False)).astype(np.float32)
    # крошечный тай-брейк, чтобы не плодить одинаковые score (при равенстве негативы идут выше)
    rng = np.random.default_rng(a.seed)
    score = score + rng.normal(scale=1e-6, size=len(score)).astype(np.float32)
    sub = pl.DataFrame({"feed_id": te["feed_id"].cast(pl.Int32), "item_id": te["item_id"].cast(pl.Int32),
                        "score": score})
    sub.write_parquet(a.out)
    print("сохранено:", a.out, sub.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict"):
        p = sp.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--pos-mode", choices=["none", "feature"], default="none")
        p.add_argument("--pos-const", type=float, default=0.0)
        p.add_argument("--max-parts", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
    t = sp.choices["train"]
    t.add_argument("--out", required=True)
    t.add_argument("--loss", default="YetiRank")
    t.add_argument("--iterations", type=int, default=2000)
    t.add_argument("--lr", type=float, default=0.05)
    t.add_argument("--depth", type=int, default=8)
    t.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    sp.choices["predict"].add_argument("--model", required=True)
    sp.choices["predict"].add_argument("--out", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
