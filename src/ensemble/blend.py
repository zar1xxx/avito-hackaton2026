"""Ансамбль моделей (метрика подбора — NDCG@30, как в эталоне): средние z-скоры внутри ленты, подбор весов и стекинг.

Вход: папки запусков с val_pred.parquet (feed_id, item_id, score) и test_pred.parquet.
Для честного ансамбля val-предсказания должны быть OOF (run_exp --mode cv для GBDT; то же — у нейросети Августа),
иначе модель обучалась на тех лентах, на которых мы подбираем веса.

  PYTHONPATH=. python -m src.ensemble.blend --data competition --runs runs/cb runs/lgb runs/nn --out runs/blend
Методы:
  mean    — равные веса
  weights — покоординатный подбор весов по pseudo-public, проверка на pseudo-private
  stack   — LightGBM lambdarank над z/rank-скорами моделей, OOF по временным блокам val
Результат: <out>/test_pred.parquet (метод --method), <out>/val_pred.parquet, <out>/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import mask_feeds, pseudo_split, time_folds
from src.common.data import load_labels
from src.common.metric import mean_ndcg
from src.common.scores import feed_rank01, feed_zscore

KEYS = ["feed_id", "item_id"]


def collect(base: pl.DataFrame, runs: list[Path], which: str) -> tuple[pl.DataFrame, list[str]]:
    names = []
    for r in runs:
        name = r.name
        names.append(name)
        p = pl.read_parquet(r / f"{which}_pred.parquet").select(KEYS + ["score"]).rename({"score": name})
        p = p.cast({"feed_id": pl.Int32, "item_id": pl.Int32})
        base = base.join(p, on=KEYS, how="left")
        miss = base[name].null_count()
        if miss:
            raise ValueError(f"{r}/{which}_pred.parquet: нет предсказаний для {miss} строк")
    return base, names


def zmatrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    f = df["feed_id"].to_numpy()
    return np.stack([feed_zscore(f, df[n].to_numpy()) for n in names], axis=1)


def rank_matrix(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    f = df["feed_id"].to_numpy()
    return np.stack([feed_rank01(f, df[n].to_numpy()) for n in names], axis=1)


def ndcg_on(df: pl.DataFrame, score: np.ndarray, feeds: np.ndarray | None = None) -> float:
    f, t = df["feed_id"].to_numpy(), df["target"].to_numpy()
    if feeds is not None:
        m = mask_feeds(f, feeds)
        return mean_ndcg(f[m], t[m], score[m], cutoff=30)
    return mean_ndcg(f, t, score, cutoff=30)


def fit_weights(df: pl.DataFrame, z: np.ndarray, feeds: np.ndarray, sweeps: int = 3) -> np.ndarray:
    """Покоординатный поиск неотрицательных весов, максимизирует NDCG на feeds."""
    w = np.ones(z.shape[1])
    best = ndcg_on(df, z @ w, feeds)
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for _ in range(sweeps):
        improved = False
        for i in range(z.shape[1]):
            for g in grid:
                cand = w.copy()
                cand[i] = g
                if cand.sum() == 0:
                    continue
                v = ndcg_on(df, z @ cand, feeds)
                if v > best + 1e-6:
                    best, w, improved = v, cand, True
        if not improved:
            break
    return w / w.sum()


def stack_features(df: pl.DataFrame, names: list[str]) -> np.ndarray:
    z, r = zmatrix(df, names), rank_matrix(df, names)
    size = df.group_by("feed_id").len().rename({"len": "n"})
    n = df.join(size, on="feed_id", how="left")["n"].to_numpy().astype(np.float32)
    return np.column_stack([z, r, z.mean(1), n])


def _lgb_fit(x: np.ndarray, y: np.ndarray, feed_id: np.ndarray, seed: int = 0):
    import lightgbm as lgb
    _, counts = np.unique(feed_id, return_counts=True)
    p = dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
             lambda_l2=5.0, seed=seed, verbose=-1, metric="ndcg", eval_at=[30])
    return lgb.train(p, lgb.Dataset(x, label=y, group=counts), num_boost_round=150)


def stack_oof(df: pl.DataFrame, names: list[str], k: int) -> np.ndarray:
    x = stack_features(df, names)
    f, y = df["feed_id"].to_numpy(), df["target"].to_numpy()
    uf, fold = time_folds(f, df["timestamp"].to_numpy(), k)
    fold_of = dict(zip(uf.tolist(), fold.tolist()))
    rf = np.array([fold_of[i] for i in f])
    oof = np.zeros(len(f))
    for j in range(k):
        tr = rf != j
        oof[~tr] = _lgb_fit(x[tr], y[tr], f[tr]).predict(x[~tr])
    return oof


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["mean", "weights", "stack"], default="weights")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in a.runs]

    lab = load_labels(a.data, "val").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    val, names = collect(lab, runs, "val")
    val = val.sort("feed_id", maintain_order=True)
    pub, priv = pseudo_split(val["feed_id"].to_numpy(), val["timestamp"].to_numpy())
    z = zmatrix(val, names)

    report: dict = {"models": {}, "blends": {}}
    for i, n in enumerate(names):
        report["models"][n] = {"pub": ndcg_on(val, z[:, i], pub), "priv": ndcg_on(val, z[:, i], priv)}

    w_mean = np.ones(len(names)) / len(names)
    w_fit = fit_weights(val, z, pub)
    scores = {"mean": z @ w_mean, "weights": z @ w_fit}
    report["weights"] = dict(zip(names, w_fit.round(4).tolist()))
    if len(names) > 1:
        scores["stack"] = stack_oof(val, names, a.folds)
    for m, s in scores.items():
        report["blends"][m] = {"pub": ndcg_on(val, s, pub), "priv": ndcg_on(val, s, priv)}
    method = a.method if a.method in scores else "mean"
    report["chosen"] = method

    # --- test ---
    te_keys = load_labels(a.data, "test").cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    te, _ = collect(te_keys, runs, "test")
    te = te.sort("feed_id", maintain_order=True)
    zt = zmatrix(te, names)
    if method == "mean":
        st = zt @ w_mean
    elif method == "weights":
        st = zt @ w_fit
    else:
        x_all = stack_features(val, names)
        model = _lgb_fit(x_all, val["target"].to_numpy(), val["feed_id"].to_numpy())
        st = model.predict(stack_features(te, names))
    pl.DataFrame({"feed_id": te["feed_id"], "item_id": te["item_id"], "score": st.astype(np.float32)}
                 ).write_parquet(out / "test_pred.parquet")
    val.select("feed_id", "item_id", "target").with_columns(pl.Series("score", scores[method])
                                                            ).write_parquet(out / "val_pred.parquet")
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{'':10s} {'pub':>8s} {'priv':>8s}")
    for n, v in report["models"].items():
        print(f"{n:10s} {v['pub']:8.4f} {v['priv']:8.4f}")
    for n, v in report["blends"].items():
        print(f"[{n}]".ljust(10) + f" {v['pub']:8.4f} {v['priv']:8.4f}" + ("  <- выбран" if n == method else ""))
    print("веса:", report["weights"])


if __name__ == "__main__":
    main()
