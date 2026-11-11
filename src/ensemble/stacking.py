"""Стекинг (мета-модель Ridge) над предсказаниями нейросети, GBDT и табличными фичами.

Принцип работы:
1. Загружает OOF-предсказания на val от нейросети (NN) и GBDT.
2. Из parquet-файлов сплита подгружает признаки кандидатов f1..f5 (без чтения эмбеддингов).
3. Обучает Ridge-регрессию (со StandardScaler) на фичах [nn_score, gbdt_score, f1, f2, f3, f4, f5]
   через GroupKFold по feed_id для честной оценки NDCG без утечек.
4. Обучает финальный Ridge на всем val и (опционально) предсказывает сабмит для test.

Пример запуска:
  python -u src/ensemble/stacking.py `
      --data /path/to/data `
      --val-nn models/nn/oof_val.parquet `
      --val-gbdt models/gbdt/oof_val.parquet `
      --test-nn models/nn/submission.parquet `
      --test-gbdt models/gbdt/submission.parquet `
      --out models/ensemble/submission_stack.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.metric import all_variants
from src.nn.dataset import FLOAT_FEATS, find_parquet_parts

KEYS = ["feed_id", "item_id"]


def load_candidate_float_feats(
    data_root: str | Path,
    split: str,
    max_parts: int | None = None,
) -> pl.DataFrame:
    """Быстро извлекает только feed_id, item_id и f1..f5 для сплита (без эмбеддингов)."""
    parts = find_parquet_parts(data_root, split)
    if max_parts is not None:
        parts = parts[:max_parts]

    dfs = []
    cols_to_read = ["feed_id", "item_id"] + FLOAT_FEATS
    for p in parts:
        df_part = pl.read_parquet(p, columns=cols_to_read)
        df_long = df_part.explode(["item_id"] + FLOAT_FEATS).cast({
            "feed_id": pl.Int32,
            "item_id": pl.Int32,
            **{f: pl.Float32 for f in FLOAT_FEATS},
        })
        dfs.append(df_long)

    return pl.concat(dfs, how="vertical")


def prepare_stack_dataframe(
    nn_path: str | Path,
    gbdt_path: str | Path,
    data_root: str | Path | None = None,
    split: str = "val",
    with_target: bool = True,
    max_parts: int | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Объединяет скоры NN, GBDT и признаки f1..f5 по (feed_id, item_id)."""
    nn_cols = KEYS + ["score"] + (["target"] if with_target else [])
    df_nn = pl.read_parquet(nn_path).select(nn_cols).rename({"score": "nn_score"})
    df_gbdt = pl.read_parquet(gbdt_path).select(KEYS + ["score"]).rename({"score": "gbdt_score"})

    df = df_nn.cast({"feed_id": pl.Int32, "item_id": pl.Int32}).join(
        df_gbdt.cast({"feed_id": pl.Int32, "item_id": pl.Int32}),
        on=KEYS,
        how="inner",
    )

    feature_cols = ["nn_score", "gbdt_score"]

    if data_root is not None:
        feats_df = load_candidate_float_feats(data_root, split, max_parts=max_parts)
        df = df.join(feats_df, on=KEYS, how="left")
        # Заполняем возможные пропуски нулями
        df = df.with_columns([pl.col(f).fill_null(0.0) for f in FLOAT_FEATS])
        feature_cols.extend(FLOAT_FEATS)

    return df, feature_cols


def build_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])


def run_stacking(args: argparse.Namespace) -> None:
    print("=" * 70)
    print(f" ЗАПУСК СТЕКИНГА (Ridge alpha={args.alpha}, GroupKFold={args.folds})")
    print("=" * 70)

    # 1. Подготовка датафрейма val
    val_df, feature_cols = prepare_stack_dataframe(
        nn_path=args.val_nn,
        gbdt_path=args.val_gbdt,
        data_root=args.data,
        split=args.val_split,
        with_target=True,
        max_parts=args.max_parts,
    )

    print(f"Строк на val: {len(val_df):,} (лент: {val_df['feed_id'].n_unique():,})")
    print(f"Признаки стекинга ({len(feature_cols)}): {feature_cols}")

    x_val = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_df["target"].to_numpy().astype(np.float32)
    feeds_val = val_df["feed_id"].to_numpy()

    # Метрики базовых моделей
    m_nn = all_variants(feeds_val, y_val, val_df["nn_score"].to_numpy())
    m_gbdt = all_variants(feeds_val, y_val, val_df["gbdt_score"].to_numpy())

    print("\n--- Базовые модели на val ---")
    print(f"  NN   : full={m_nn['ndcg_full']:.4f} | IDCG@30={m_nn['ndcg_idcg30']:.4f} | @30={m_nn['ndcg@30']:.4f}")
    print(f"  GBDT : full={m_gbdt['ndcg_full']:.4f} | IDCG@30={m_gbdt['ndcg_idcg30']:.4f} | @30={m_gbdt['ndcg@30']:.4f}")

    # 2. Честный OOF через GroupKFold по feed_id
    n_unique_feeds = val_df["feed_id"].n_unique()
    n_splits = min(args.folds, n_unique_feeds)
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(val_df), dtype=np.float32)
    for train_idx, test_idx in gkf.split(x_val, y_val, groups=feeds_val):
        pipe = build_ridge_pipeline(alpha=args.alpha)
        pipe.fit(x_val[train_idx], y_val[train_idx])
        oof_preds[test_idx] = pipe.predict(x_val[test_idx]).astype(np.float32)

    m_stack = all_variants(feeds_val, y_val, oof_preds)
    print("-" * 70)
    print(f"  [STACK OOF] Ridge: full={m_stack['ndcg_full']:.4f} | IDCG@30={m_stack['ndcg_idcg30']:.4f} | @30={m_stack['ndcg@30']:.4f}")
    print("-" * 70)

    # 3. Финальное обучение Ridge на всем val и вывод весов
    final_pipe = build_ridge_pipeline(alpha=args.alpha)
    final_pipe.fit(x_val, y_val)

    coefs = final_pipe.named_steps["ridge"].coef_
    print("\nВеса признаков в Ridge (после StandardScaler):")
    for fname, coef in zip(feature_cols, coefs):
        print(f"  {fname:12s}: {coef:+.5f}")

    # Сохраняем OOF стекинга при необходимости
    if args.out_oof:
        oof_out = val_df.select(KEYS + ["target"]).with_columns(
            pl.Series("score", oof_preds, dtype=pl.Float32)
        )
        Path(args.out_oof).parent.mkdir(parents=True, exist_ok=True)
        oof_out.write_parquet(args.out_oof)
        print(f"[OK] OOF стекинга сохранен: {args.out_oof}")

    # 4. Инференс на test (если переданы пути к test-предсказаниям)
    if args.test_nn and args.test_gbdt:
        print("\n--- Генерация test сабмита стекингом ---")
        test_df, test_feat_cols = prepare_stack_dataframe(
            nn_path=args.test_nn,
            gbdt_path=args.test_gbdt,
            data_root=args.data,
            split=args.test_split,
            with_target=False,
            max_parts=args.max_parts,
        )
        assert test_feat_cols == feature_cols, "Наборы фичей на val и test не совпадают!"

        x_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
        test_preds = final_pipe.predict(x_test).astype(np.float32)

        sub = test_df.select(KEYS).with_columns(
            pl.Series("score", test_preds, dtype=pl.Float32)
        ).cast({"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32})

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(out_path)
        print(f"[OK] Финальный сабмит стекинга сохранен: {out_path} (строк: {len(sub):,})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Стекинг Ridge над NN, GBDT и фичами f1..f5")
    parser.add_argument("--data", default=None, help="Путь к корню данных для загрузки f1..f5 (если не указан — только скоры)")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--val-nn", required=True, help="OOF-предсказания NN на val")
    parser.add_argument("--val-gbdt", required=True, help="OOF-предсказания GBDT на val")
    parser.add_argument("--test-nn", default=None, help="Предсказания NN на test")
    parser.add_argument("--test-gbdt", default=None, help="Предсказания GBDT на test")
    parser.add_argument("--alpha", type=float, default=10.0, help="L2-регуляризация Ridge")
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов GroupKFold по feed_id")
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--out", default="models/ensemble/submission_stack.parquet", help="Выходной сабмит")
    parser.add_argument("--out-oof", default=None, help="Куда сохранить OOF предсказания стекинга (опционально)")
    args = parser.parse_args()

    run_stacking(args)


if __name__ == "__main__":
    main()
