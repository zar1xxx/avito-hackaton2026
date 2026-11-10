"""Раннер экспериментов GBDT.

Режимы:
  holdout — обучение на train, early stopping на pseudo-public части val, метрики на val и test-предсказания
  cv      — K временных блоков val: каждый блок предсказывается моделью, обученной на train + остальной val
            (даёт OOF-предсказания на всём val для ансамбля/стекинга; итерации фиксированы)
  final   — обучение на train + весь val, предсказание test (метрик нет: val в обучении)

Работа с позицией (--pos-mode):
  none     — pos не используется
  feature  — pos как признак; на val/test подставляется --pos-const
  ipw      — веса позитивов 1/p(pos) (см. weights.py)
  topn     — обучаться только на строках train с pos < --topn (меньше смещение, меньше данных)

Пример:
  PYTHONPATH=. python -m src.gbdt.run_exp --data competition --out runs/cb-yeti --kind catboost
  PYTHONPATH=. python -m src.gbdt.run_exp --data competition --out runs/lgb-ipw --kind lightgbm --pos-mode ipw
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import MODEL_FEATURES, load_split
from src.common.metric import all_variants
from src.common.scores import feed_zscore
from src.gbdt.extra_features import attach
from src.gbdt.models import Ranker
from src.gbdt.weights import ipw_weights, position_ctr


def get_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["holdout", "cv", "final"], default="holdout")
    ap.add_argument("--kind", choices=["catboost", "lightgbm"], default="catboost")
    ap.add_argument("--loss", default=None, help="YetiRank / YetiRankPairwise / QueryRMSE / lambdarank ...")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--iterations-from", default=None, help="metrics.json holdout-запуска: взять best_iteration")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--pos-mode", choices=["none", "feature", "ipw", "topn"], default="none")
    ap.add_argument("--pos-const", type=float, default=0.0)
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--ipw-max", type=float, default=10.0)
    ap.add_argument("--val-weight", type=float, default=1.0, help="вес строк val при обучении на train+val")
    ap.add_argument("--folds", type=int, default=4, help="число временных блоков для режима cv")
    ap.add_argument("--model-params", default=None, help="JSON-файл с доп. параметрами модели (пишет tune.py)")
    ap.add_argument("--features-dir", default=None, help="папка с внешними признаками (extra_features.py)")
    ap.add_argument("--extra", nargs="*", default=[], help="группы внешних признаков, напр. anton avgust")
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=1, help="процессов при построении признаков")
    ap.add_argument("--cache", default=None, help="папка кэша таблиц признаков")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--log", action="store_true", help="дописать строку в experiments/log.md (только для реальных данных)")
    return ap.parse_args(argv)


def load_all(a: argparse.Namespace, need_test: bool) -> tuple[dict[str, pl.DataFrame], list[str]]:
    frames, extra_cols = {}, []
    splits = ["train", "val"] + (["test"] if need_test else [])
    for s in splits:
        df = load_split(a.data, s, a.max_parts, a.n_jobs, a.cache)
        df, extra_cols = attach(df, a.features_dir, a.extra, s)
        if "pos" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Int16).alias("pos"))
        frames[s] = df.with_columns(pl.col("category_id").cast(pl.Int32), pl.col("pos").cast(pl.Int16)
                                    ).sort("feed_id", maintain_order=True)
    return frames, MODEL_FEATURES + extra_cols


def apply_pos(df: pl.DataFrame, a: argparse.Namespace, is_train: bool) -> pl.DataFrame:
    """Для режима feature: pos в train — настоящий (null -> -1), в val/test — константа."""
    if a.pos_mode != "feature":
        return df
    if is_train:
        return df.with_columns(pl.col("pos").fill_null(-1).cast(pl.Float32))
    return df.with_columns(pl.lit(a.pos_const).cast(pl.Float32).alias("pos"))


def train_frame(frames: dict[str, pl.DataFrame], a: argparse.Namespace, val_rows: pl.DataFrame | None
                ) -> tuple[pl.DataFrame, np.ndarray | None]:
    """Обучающая таблица: train (с учётом режима позиции) [+ часть val] и веса строк."""
    tr = frames["train"]
    w = None
    if a.pos_mode == "topn":
        tr = tr.filter(pl.col("pos").is_null() | (pl.col("pos") < a.topn))
    elif a.pos_mode == "ipw":
        ctr = position_ctr(frames["train"]["pos"].to_numpy(), frames["train"]["target"].to_numpy())
        w = ipw_weights(tr["pos"].to_numpy(), tr["target"].to_numpy(), ctr, a.ipw_max)
    tr = apply_pos(tr, a, True)
    if val_rows is not None and val_rows.height:
        v = apply_pos(val_rows, a, False)
        tr = pl.concat([tr, v.select(tr.columns)], how="vertical")
        if w is not None or a.val_weight != 1.0:
            w0 = w if w is not None else np.ones(tr.height - v.height, dtype=np.float32)
            w = np.concatenate([w0, np.full(v.height, a.val_weight, dtype=np.float32)])
    return tr, w


def make_ranker(a: argparse.Namespace, feats: list[str], iterations: int | None = None) -> Ranker:
    feats = feats + (["pos"] if a.pos_mode == "feature" else [])
    extra = json.loads(Path(a.model_params).read_text()) if a.model_params else None
    return Ranker(a.kind, feats, a.loss, iterations or a.iterations, a.lr, a.depth, a.seed, a.task_type, extra)


def split_metrics(va: pl.DataFrame, score: np.ndarray, pub: np.ndarray, priv: np.ndarray) -> dict:
    f, t = va["feed_id"].to_numpy(), va["target"].to_numpy()
    out = {}
    for name, ids in (("all", None), ("pub", pub), ("priv", priv)):
        m = np.ones(len(f), bool) if ids is None else mask_feeds(f, ids)
        if m.any():
            for k, v in all_variants(f[m], t[m], score[m]).items():
                out[f"{name}_{k}"] = v
    return out


def append_log(a: argparse.Namespace, res: dict) -> None:
    log = Path(__file__).resolve().parents[2] / "experiments" / "log.md"
    if not log.exists():
        return
    desc = (f"{a.kind}/{a.loss or 'default'} pos={a.pos_mode} mode={a.mode} extra={','.join(a.extra) or '-'} "
            f"depth={a.depth} lr={a.lr} {a.note}").strip()
    v = "/".join(f"{res.get(k, float('nan')):.4f}" for k in
                 ("priv_ndcg_full", "priv_ndcg_idcg30", "priv_ndcg@30"))
    with log.open("a") as fh:
        fh.write(f"| {dt.date.today()} | danya | - | {desc} | priv: {v} | - | out={a.out} |\n")


def run(a: argparse.Namespace) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    need_test = (not a.no_test) or a.mode == "final"
    frames, feats = load_all(a, need_test)
    va_raw = frames["val"]
    pub, priv = pseudo_split(va_raw["feed_id"].to_numpy(), va_raw["timestamp"].to_numpy())
    iters = a.iterations
    if a.iterations_from:
        iters = int(json.loads(Path(a.iterations_from).read_text())["best_iteration"])
    res: dict = {"args": vars(a), "n_features": len(feats)}
    test_scores: list[np.ndarray] = []

    if a.mode == "holdout":
        tr, w = train_frame(frames, a, None)
        va = apply_pos(va_raw, a, False)
        es = va.filter(pl.col("feed_id").is_in(pub))
        model = make_ranker(a, feats, iters).fit(tr, w, es if es.height else va)
        score = model.predict(va)
        res.update(split_metrics(va, score, pub, priv))
        res["best_iteration"] = model.best_iteration
        val_pred = va.select("feed_id", "item_id", "target").with_columns(pl.Series("score", score))
        if need_test:
            test_scores.append(model.predict(apply_pos(frames["test"], a, False)))

    elif a.mode == "cv":
        uf, fold = time_folds(va_raw["feed_id"].to_numpy(), va_raw["timestamp"].to_numpy(), a.folds)
        fold_of = dict(zip(uf.tolist(), fold.tolist()))
        row_fold = np.array([fold_of[x] for x in va_raw["feed_id"].to_numpy()])
        oof = np.zeros(va_raw.height)
        for k in range(a.folds):
            held = row_fold == k
            tr, w = train_frame(frames, a, va_raw.filter(pl.Series(~held)))
            model = make_ranker(a, feats, iters).fit(tr, w)
            oof[held] = model.predict(apply_pos(va_raw.filter(pl.Series(held)), a, False))
            if need_test:
                te = apply_pos(frames["test"], a, False)
                test_scores.append(feed_zscore(te["feed_id"].to_numpy(), model.predict(te)))
        va = apply_pos(va_raw, a, False)
        res.update(split_metrics(va, oof, pub, priv))
        val_pred = va.select("feed_id", "item_id", "target").with_columns(pl.Series("score", oof))

    else:  # final
        tr, w = train_frame(frames, a, va_raw)
        model = make_ranker(a, feats, iters).fit(tr, w)
        test_scores.append(model.predict(apply_pos(frames["test"], a, False)))
        val_pred = None

    if need_test and test_scores:
        te = frames["test"]
        s = np.mean(test_scores, axis=0).astype(np.float32)
        sub = pl.DataFrame({"feed_id": te["feed_id"].cast(pl.Int32), "item_id": te["item_id"].cast(pl.Int32),
                            "score": s})
        sub.write_parquet(out / "test_pred.parquet")
    if val_pred is not None:
        val_pred.write_parquet(out / "val_pred.parquet")
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    if a.mode != "final":
        if a.log:
            append_log(a, res)
        print({k: round(v, 4) for k, v in res.items() if k.endswith(("ndcg_full", "ndcg@30"))})
    return res


if __name__ == "__main__":
    run(get_args())
