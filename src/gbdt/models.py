"""Единая обёртка над CatBoost и LightGBM для ранжирования (группа = лента)."""
from __future__ import annotations

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

CAT = "category_id"


def _x(df: pl.DataFrame, feats: list[str]):
    return df.select(feats).to_pandas()


def _group_sizes(feed_id: np.ndarray) -> np.ndarray:
    # данные отсортированы по feed_id, ленты идут подряд
    _, counts = np.unique(feed_id, return_counts=True)
    return counts


class Ranker:
    def __init__(self, kind: str, feats: list[str], loss: str | None = None, iterations: int = 2000,
                 lr: float = 0.05, depth: int = 8, seed: int = 0, task_type: str = "CPU",
                 extra: dict | None = None):
        self.kind, self.feats, self.seed = kind, feats, seed
        self.iterations, self.lr, self.depth = iterations, lr, depth
        self.loss = loss or ("YetiRank" if kind == "catboost" else "lambdarank")
        self.task_type, self.extra = task_type, extra or {}
        self.model = None
        self.best_iteration: int | None = None

    # ---- обучение -------------------------------------------------------------------------
    def fit(self, tr: pl.DataFrame, weight: np.ndarray | None = None, va: pl.DataFrame | None = None,
            es_rounds: int = 100) -> "Ranker":
        if weight is not None and self.kind == "catboost" and self.loss.startswith(("YetiRank", "PairLogit")):
            # CatBoost для парных лоссов только пишет предупреждение и ИГНОРИРУЕТ веса объектов
            raise ValueError(f"CatBoost {self.loss} не поддерживает веса объектов (IPW/val-weight). "
                             "Используйте --kind lightgbm или --loss LambdaMart / QueryRMSE.")
        return self._fit_cb(tr, weight, va, es_rounds) if self.kind == "catboost" \
            else self._fit_lgb(tr, weight, va, es_rounds)

    def _fit_cb(self, tr, weight, va, es_rounds):
        params = dict(loss_function=self.loss, iterations=self.iterations, learning_rate=self.lr,
                      depth=self.depth, eval_metric="NDCG", random_seed=self.seed,
                      task_type=self.task_type, verbose=200, allow_writing_files=False, **self.extra)
        if va is not None:
            params.update(od_type="Iter", od_wait=es_rounds)
        m = CatBoostRanker(**params)
        pool = Pool(_x(tr, self.feats), label=tr["target"].to_numpy(), group_id=tr["feed_id"].to_numpy(),
                    cat_features=[CAT], weight=weight)
        ev = None
        if va is not None:
            ev = Pool(_x(va, self.feats), label=va["target"].to_numpy(), group_id=va["feed_id"].to_numpy(),
                      cat_features=[CAT])
        m.fit(pool, eval_set=ev, use_best_model=va is not None)
        self.model = m
        self.best_iteration = int(m.get_best_iteration()) + 1 if va is not None else self.iterations
        return self

    def _fit_lgb(self, tr, weight, va, es_rounds):
        import lightgbm as lgb
        params = dict(objective=self.loss, learning_rate=self.lr, num_leaves=2 ** min(self.depth, 8) - 1,
                      max_depth=-1, metric="ndcg", eval_at=[30], seed=self.seed, verbose=-1,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, **self.extra)
        dtr = lgb.Dataset(_x(tr, self.feats), label=tr["target"].to_numpy(), weight=weight,
                          group=_group_sizes(tr["feed_id"].to_numpy()), categorical_feature=[CAT])
        cbs = [lgb.log_evaluation(200)]
        valid = []
        if va is not None:
            valid = [lgb.Dataset(_x(va, self.feats), label=va["target"].to_numpy(),
                                 group=_group_sizes(va["feed_id"].to_numpy()), categorical_feature=[CAT],
                                 reference=dtr)]
            cbs.append(lgb.early_stopping(es_rounds, verbose=False))
        self.model = lgb.train(params, dtr, num_boost_round=self.iterations, valid_sets=valid, callbacks=cbs)
        self.best_iteration = int(self.model.best_iteration or self.iterations) if va is not None \
            else self.iterations
        return self

    # ---- предсказание ---------------------------------------------------------------------
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.kind == "catboost":
            pool = Pool(_x(df, self.feats), group_id=df["feed_id"].to_numpy(), cat_features=[CAT])
            return np.asarray(self.model.predict(pool), dtype=np.float64)
        return np.asarray(self.model.predict(_x(df, self.feats), num_iteration=self.best_iteration),
                          dtype=np.float64)
