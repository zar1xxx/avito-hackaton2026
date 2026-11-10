"""Подбор гиперпараметров (Optuna) по val.

Схема: обучение на train, early stopping по pseudo-public части val, цель — NDCG на pseudo-private.
Так подбор не видит ту часть, по которой считается цель, но много попыток всё равно «подгоняют» под val:
лучшую конфигурацию надо перепроверить режимом cv (run_exp --mode cv) и сравнить с базовой.

    PYTHONPATH=. python -m src.gbdt.tune --data competition --out runs/tune-cb --kind catboost --trials 30
Лучшие параметры -> <out>/best_params.json, extra_params.json; запуск: run_exp.py ... $(cat <out>/best_cli.txt)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl

from src.common.cv import mask_feeds, pseudo_split
from src.common.metric import mean_ndcg
from src.gbdt.models import Ranker
from src.gbdt.run_exp import apply_pos, get_args, load_all, train_frame


def suggest(trial: optuna.Trial, kind: str) -> tuple[dict, dict]:
    """Возвращает (основные параметры, extra-параметры модели)."""
    main = dict(depth=trial.suggest_int("depth", 4, 10), lr=trial.suggest_float("lr", 0.02, 0.15, log=True))
    if kind == "catboost":
        extra = dict(l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                     random_strength=trial.suggest_float("random_strength", 0.1, 3.0, log=True))
    else:
        extra = dict(min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
                     lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
                     min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0))
    return main, extra


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None, help="секунд на весь подбор")
    ns, rest = ap.parse_known_args(argv)
    a = get_args(rest + ["--no-test"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, feats = load_all(a, need_test=False)
    va_raw = frames["val"]
    pub, priv = pseudo_split(va_raw["feed_id"].to_numpy(), va_raw["timestamp"].to_numpy())
    va = apply_pos(va_raw, a, False)
    es = va.filter(pl.col("feed_id").is_in(pub))
    ev = va.filter(pl.col("feed_id").is_in(priv))
    tr, w = train_frame(frames, a, None)
    feats = feats + (["pos"] if a.pos_mode == "feature" else [])

    def objective(trial: optuna.Trial) -> float:
        main_p, extra = suggest(trial, a.kind)
        m = Ranker(a.kind, feats, a.loss, a.iterations, main_p["lr"], main_p["depth"], a.seed, a.task_type, extra)
        m.fit(tr, w, es)
        trial.set_user_attr("best_iteration", m.best_iteration)
        return mean_ndcg(ev["feed_id"].to_numpy(), ev["target"].to_numpy(), m.predict(ev))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
    study.optimize(objective, n_trials=ns.trials, timeout=ns.timeout)
    best = study.best_trial
    (out / "best_params.json").write_text(json.dumps(
        {"value": best.value, "params": best.params, "best_iteration": best.user_attrs["best_iteration"]}, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    extra = {k: v for k, v in best.params.items() if k not in ("depth", "lr")}
    (out / "extra_params.json").write_text(json.dumps(extra, indent=2))
    cli = (f"--depth {best.params['depth']} --lr {best.params['lr']:.5f} "
           f"--iterations {best.user_attrs['best_iteration']} --model-params {out / 'extra_params.json'}")
    (out / "best_cli.txt").write_text(cli)
    print("лучший priv NDCG:", round(best.value, 5), "|", best.params, "| iters:", best.user_attrs["best_iteration"])
    print("запуск с лучшими параметрами: run_exp.py ... " + cli)


if __name__ == "__main__":
    main()
