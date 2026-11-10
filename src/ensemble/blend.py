"""Ранговый блендинг предсказаний нейросети и GBDT.

Принцип работы:
1. Скоры каждой модели внутри ленты (feed_id) переводятся в нормализованные ранги [0..1]
   (1.0 — кандидат с наивысшим скором, 0.0 — с наименьшим).
2. Итоговый скор: final_score = w_nn * rank_nn + (1 - w_nn) * rank_gbdt.
3. По сетке w_nn in [0.00, 0.05, ..., 1.00] подбирается вес, максимизирующий NDCG на val.
4. (Опционально) Веса применяются к test-предсказаниям моделей для создания финального сабмита.

Инструкция для Дани:
  # 1. Поиск оптимальных весов на val:
  python -m src.ensemble.blend `
      --val-nn runs/nn/oof_val.parquet `
      --val-gbdt runs/gbdt/oof_val.parquet

  # 2. Поиск весов + генерация финального сабмита:
  python -m src.ensemble.blend `
      --val-nn runs/nn/oof_val.parquet `
      --val-gbdt runs/gbdt/oof_val.parquet `
      --test-nn runs/nn/test_pred.parquet `
      --test-gbdt runs/gbdt/test_pred.parquet `
      --out runs/ensemble/submission_blend.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl

from src.common.metric import all_variants, mean_ndcg

KEYS = ["feed_id", "item_id"]


def compute_feed_ranks(df: pl.DataFrame, score_col: str, out_col: str) -> pl.DataFrame:
    """Вычисляет нормализованный ранг кандидата внутри ленты [0..1].
    
    1.0 = лучший кандидат (наибольший score).
    0.0 = худший кандидат (наименьший score).
    """
    # ordinal rank: 1 для наименьшего, M для наибольшего
    return df.with_columns(
        (
            (pl.col(score_col).rank("ordinal").over("feed_id") - 1.0)
            / pl.max_horizontal(pl.len().over("feed_id") - 1.0, pl.lit(1.0))
        ).cast(pl.Float32).alias(out_col)
    )


def blend_val(
    val_nn_path: str | Path,
    val_gbdt_path: str | Path,
    step: float = 0.05,
    metric_name: str = "ndcg_full",
) -> tuple[float, float, dict[str, float]]:
    """Находит оптимальный вес w_nn по сетке, максимизирующий NDCG на val.
    
    Returns:
        best_w: лучший вес нейросети w_nn in [0, 1]
        best_metric: значение NDCG при лучшем весе
        report: словарь метрик лучших моделей и ансамбля
    """
    print(f"Чтение NN OOF:   {val_nn_path}")
    print(f"Чтение GBDT OOF: {val_gbdt_path}")

    df_nn = pl.read_parquet(val_nn_path).select(KEYS + ["score", "target"]).rename({"score": "score_nn"})
    df_gbdt = pl.read_parquet(val_gbdt_path).select(KEYS + ["score"]).rename({"score": "score_gbdt"})

    # Объединяем по (feed_id, item_id)
    val = df_nn.join(df_gbdt, on=KEYS, how="inner")
    if len(val) == 0:
        raise ValueError("Пересечение по (feed_id, item_id) пусто! Проверьте валидационные файлы.")

    print(f"Общих валидационных кандидатов: {len(val):,} (лент: {val['feed_id'].n_unique():,})")

    # Нормализованные ранги внутри лент
    val = compute_feed_ranks(val, "score_nn", "rank_nn")
    val = compute_feed_ranks(val, "score_gbdt", "rank_gbdt")

    feed_ids = val["feed_id"].to_numpy()
    targets = val["target"].to_numpy()
    rank_nn = val["rank_nn"].to_numpy()
    rank_gbdt = val["rank_gbdt"].to_numpy()

    # Считаем NDCG для отдельных моделей
    metrics_nn = all_variants(feed_ids, targets, rank_nn)
    metrics_gbdt = all_variants(feed_ids, targets, rank_gbdt)

    print("\n" + "=" * 65)
    print(" МЕТРИКИ ОДИНОЧНЫХ МОДЕЛЕЙ НА VAL ")
    print("=" * 65)
    print(f"NN   (w=1.0) : NDCG_full={metrics_nn['ndcg_full']:.4f} | IDCG@30={metrics_nn['ndcg_idcg30']:.4f} | @30={metrics_nn['ndcg@30']:.4f}")
    print(f"GBDT (w=0.0) : NDCG_full={metrics_gbdt['ndcg_full']:.4f} | IDCG@30={metrics_gbdt['ndcg_idcg30']:.4f} | @30={metrics_gbdt['ndcg@30']:.4f}")
    print("-" * 65)

    # Сетка поиска весов
    weights = np.arange(0.0, 1.0 + 1e-5, step)
    best_w = 0.5
    best_score = -1.0
    best_metrics = {}

    print(f"Поиск весов w_nn in [0.0..1.0] с шагом {step:.2f} (критерий: {metric_name}):")

    for w in weights:
        w_nn = float(w)
        w_gbdt = 1.0 - w_nn
        blend_rank = w_nn * rank_nn + w_gbdt * rank_gbdt

        m = all_variants(feed_ids, targets, blend_rank)
        score = m[metric_name]

        if score > best_score:
            best_score = score
            best_w = w_nn
            best_metrics = m

        print(f"  w_nn={w_nn:.2f} | w_gbdt={w_gbdt:.2f} -> {metric_name}={score:.4f} (full={m['ndcg_full']:.4f}, @30={m['ndcg@30']:.4f})")

    print("=" * 65)
    gain_nn = best_score - metrics_nn[metric_name]
    gain_gbdt = best_score - metrics_gbdt[metric_name]
    print(f" ЛУЧШИЙ АНСАМБЛЬ: w_nn={best_w:.2f}, w_gbdt={1.0 - best_w:.2f}")
    print(f" Лучший {metric_name}: {best_score:.4f} (прирост к NN: {gain_nn:+.4f}, к GBDT: {gain_gbdt:+.4f})")
    print(f" Итоговые метрики ансамбля: full={best_metrics['ndcg_full']:.4f} | IDCG@30={best_metrics['ndcg_idcg30']:.4f} | @30={best_metrics['ndcg@30']:.4f}")
    print("=" * 65)

    return best_w, best_score, best_metrics


def blend_test(
    test_nn_path: str | Path,
    test_gbdt_path: str | Path,
    w_nn: float,
    out_path: str | Path,
) -> None:
    """Применяет подобранный вес к предсказаниям на test и сохраняет сабмит."""
    print(f"\nГенерация финального сабмита: w_nn={w_nn:.2f}, w_gbdt={1.0 - w_nn:.2f}")
    print(f"Test NN:   {test_nn_path}")
    print(f"Test GBDT: {test_gbdt_path}")

    te_nn = pl.read_parquet(test_nn_path).select(KEYS + ["score"]).rename({"score": "score_nn"})
    te_gbdt = pl.read_parquet(test_gbdt_path).select(KEYS + ["score"]).rename({"score": "score_gbdt"})

    te = te_nn.join(te_gbdt, on=KEYS, how="inner")
    if len(te) != len(te_nn):
        print(f"Внимание: пересечение test строк ({len(te):,}) отличается от NN ({len(te_nn):,})")

    te = compute_feed_ranks(te, "score_nn", "rank_nn")
    te = compute_feed_ranks(te, "score_gbdt", "rank_gbdt")

    w_gbdt = 1.0 - w_nn
    sub = te.with_columns(
        (pl.col("rank_nn") * w_nn + pl.col("rank_gbdt") * w_gbdt).cast(pl.Float32).alias("score")
    ).select(["feed_id", "item_id", "score"])

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sub.cast({"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}).write_parquet(out_file)
    print(f"[OK] Сабмит ансамбля сохранен в: {out_file} (строк: {len(sub):,})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ранговый блендинг предсказаний NN и GBDT")
    parser.add_argument("--val-nn", required=True, help="Путь к OOF-предсказаниям нейросети на val")
    parser.add_argument("--val-gbdt", required=True, help="Путь к OOF-предсказаниям GBDT на val")
    parser.add_argument("--step", type=float, default=0.05, help="Шаг сетки подбора весов")
    parser.add_argument("--metric", choices=["ndcg_full", "ndcg_idcg30", "ndcg@30"], default="ndcg_full")
    parser.add_argument("--test-nn", default=None, help="Путь к предсказаниям нейросети на test (опционально)")
    parser.add_argument("--test-gbdt", default=None, help="Путь к предсказаниям GBDT на test (опционально)")
    parser.add_argument("--out", default="models/ensemble/submission_blend.parquet", help="Путь для сохранения сабмита")
    args = parser.parse_args()

    best_w, _, _ = blend_val(
        val_nn_path=args.val_nn,
        val_gbdt_path=args.val_gbdt,
        step=args.step,
        metric_name=args.metric,
    )

    if args.test_nn is not None and args.test_gbdt is not None:
        blend_test(
            test_nn_path=args.test_nn,
            test_gbdt_path=args.test_gbdt,
            w_nn=best_w,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
