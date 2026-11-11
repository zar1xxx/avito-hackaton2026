"""Обучение Weighted Seed Ensemble (взвешенный ансамбль нейросетей по разным сидам).

Последовательно обучает K моделей с разными сидами, затем:
1. Прогоняет каждую модель по val-выборке.
2. Подбирает оптимальные веса моделей по метрике NDCG (координатный спуск + проверка вершин).
   Благодаря этому слабые сиды получают меньший вес, а сильные — больший.
3. Сохраняет OOF-предсказания ансамбля и (при --predict-test) финальный сабмит с найденными весами.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.common.metric import all_variants, mean_ndcg
from src.nn.dataset import FeedRankingDataset, collate_ranking_batch
from src.nn.predict import (
    generate_predictions,
    normalize_feed_ranks,
    normalize_feed_zscore,
    predict_single_model,
)


def optimize_ensemble_weights(
    feed_ids: np.ndarray,
    targets: np.ndarray,
    normed_scores: list[np.ndarray],
    n_sweeps: int = 4,
    metric_cutoff: int | None = None,
) -> tuple[list[float], float]:
    """Подбирает неотрицательные веса на симплексе, максимизируя Val NDCG.
    
    Включает проверку каждого отдельного сида, равномерных весов и координатный поиск.
    """
    k = len(normed_scores)
    stacked = np.stack(normed_scores, axis=0)  # [K, N]

    def eval_w(w_arr: np.ndarray) -> float:
        w_norm = w_arr / max(float(w_arr.sum()), 1e-12)
        blend = np.tensordot(w_norm, stacked, axes=([0], [0]))
        return mean_ndcg(feed_ids, targets, blend, cutoff=metric_cutoff)

    # 1. Стартуем с лучшего из: равные веса или одиночные модели
    candidates = [np.ones(k, dtype=np.float64) / k]
    for i in range(k):
        e = np.zeros(k, dtype=np.float64)
        e[i] = 1.0
        candidates.append(e)

    best_w = candidates[0]
    best_score = -1.0
    for c in candidates:
        s = eval_w(c)
        if s > best_score:
            best_score = s
            best_w = c.copy()

    # 2. Координатный поиск по сетке коэффициентов
    grid = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    for _ in range(n_sweeps):
        improved = False
        for i in range(k):
            for g in grid:
                cand = best_w.copy()
                cand[i] = g
                if cand.sum() <= 0:
                    continue
                s = eval_w(cand)
                if s > best_score + 1e-6:
                    best_score = s
                    best_w = cand
                    improved = True
        if not improved:
            break

    best_w = best_w / best_w.sum()
    return [round(float(x), 4) for x in best_w], float(best_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение Weighted Seed Ensemble для нейросети Avito")
    parser.add_argument("--data", required=True, help="Путь к корню данных")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Список сидов")
    parser.add_argument("--out", default="models/nn/seed_ensemble", help="Базовая папка ансамбля")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--max-feeds", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--agg", choices=["rank", "zscore", "mean"], default="rank")
    parser.add_argument("--predict-test", action="store_true", help="Сгенерировать сабмит на test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 75)
    print(f" ЗАПУСК WEIGHTED SEED ENSEMBLE ({len(args.seeds)} модели: seeds={args.seeds})")
    print(f" Данные: {args.data} | Папка: {out_root}")
    print("=" * 75)

    checkpoints: list[Path] = []
    t_start = time.perf_counter()

    for idx, seed in enumerate(args.seeds, start=1):
        seed_dir = out_root / f"seed_{seed}"
        print(f"\n>>> [{idx}/{len(args.seeds)}] Обучение модели с seed={seed} -> {seed_dir}")

        cmd = [
            sys.executable, "-u", str(ROOT / "src" / "nn" / "train.py"),
            "--data", str(args.data),
            "--out", str(seed_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--dropout", str(args.dropout),
            "--hidden-dim", str(args.hidden_dim),
            "--mixup-alpha", str(args.mixup_alpha),
            "--temperature", str(args.temperature),
            "--num-workers", str(args.num_workers),
        ]
        if args.max_parts is not None:
            cmd.extend(["--max-parts", str(args.max_parts)])
        if args.max_feeds is not None:
            cmd.extend(["--max-feeds", str(args.max_feeds)])
        if args.device is not None:
            cmd.extend(["--device", args.device])

        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Ошибка при обучении модели с seed={seed} (код {res.returncode})")

        best_ckpt = seed_dir / "best_model.pt"
        if not best_ckpt.exists():
            best_ckpt = seed_dir / "last_model.pt"
        checkpoints.append(best_ckpt)

    total_train_time = time.perf_counter() - t_start
    print("\n" + "=" * 75)
    print(f" Обучение всех {len(checkpoints)} моделей завершено за {total_train_time:.1f}с")
    print("=" * 75)

    # 1. Прогон всех моделей по val для подбора оптимальных весов
    val_ds = FeedRankingDataset(
        data_root=args.data,
        split="val",
        max_parts=args.max_parts,
        max_feeds=args.max_feeds // 3 if args.max_feeds else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ranking_batch)

    feed_ids, item_ids, targets = None, None, None
    raw_scores_list = []

    print("\n--- Оценка моделей на val и подбор весов ---")
    for i, ckpt_path in enumerate(checkpoints):
        f_ids, i_ids, scores, tgts = predict_single_model(ckpt_path, val_loader, device)
        if feed_ids is None:
            feed_ids, item_ids, targets = f_ids, i_ids, tgts
        raw_scores_list.append(scores)
        m = all_variants(feed_ids, targets, scores)
        print(f"  Seed {args.seeds[i]}: NDCG_full={m['ndcg_full']:.4f} | @30={m['ndcg@30']:.4f}")

    if args.agg == "rank":
        normed = [normalize_feed_ranks(feed_ids, s) for s in raw_scores_list]
    elif args.agg == "zscore":
        normed = [normalize_feed_zscore(feed_ids, s) for s in raw_scores_list]
    else:
        normed = raw_scores_list

    # Подбираем оптимальные веса
    best_weights, best_ndcg = optimize_ensemble_weights(feed_ids, targets, normed)
    equal_ndcg = mean_ndcg(feed_ids, targets, np.mean(np.stack(normed, axis=0), axis=0))

    print("-" * 75)
    print(f"Равномерный ансамбль (equal): Val NDCG_full = {equal_ndcg:.4f}")
    print(f"Взвешенный ансамбль (weighted): Val NDCG_full = {best_ndcg:.4f}")
    print(f"Подобранные веса (seeds {args.seeds}): {best_weights}")
    print("-" * 75)

    # Сохраняем отчет о весах
    weights_report = {
        "seeds": args.seeds,
        "checkpoints": [str(p) for p in checkpoints],
        "weights": best_weights,
        "agg": args.agg,
        "val_ndcg_equal": round(equal_ndcg, 5),
        "val_ndcg_weighted": round(best_ndcg, 5),
    }
    (out_root / "ensemble_weights.json").write_text(json.dumps(weights_report, indent=2), encoding="utf-8")

    # Сохраняем взвешенный OOF для последующего стекинга / блендинга
    w_arr = np.asarray(best_weights, dtype=np.float32)
    oof_scores = np.tensordot(w_arr, np.stack(normed, axis=0), axes=([0], [0])).astype(np.float32)
    oof_df = pl.DataFrame({
        "feed_id": pl.Series(feed_ids, dtype=pl.Int32),
        "item_id": pl.Series(item_ids, dtype=pl.Int32),
        "score": pl.Series(oof_scores, dtype=pl.Float32),
        "target": pl.Series(targets, dtype=pl.Int8),
    })
    oof_path = out_root / "oof_val_ensemble.parquet"
    oof_df.write_parquet(oof_path)
    print(f"[OK] Взвешенный OOF сохранен: {oof_path}")

    # Опциональная генерация test-сабмита с подобранными весами
    if args.predict_test:
        sub_path = out_root / "submission_ensemble.parquet"
        generate_predictions(
            data_root=args.data,
            checkpoint_paths=checkpoints,
            split="test",
            batch_size=args.batch_size,
            agg=args.agg,
            weights=best_weights,
            max_parts=args.max_parts,
            max_feeds=args.max_feeds,
            device_str=args.device,
            out_path=str(sub_path),
        )


if __name__ == "__main__":
    main()
