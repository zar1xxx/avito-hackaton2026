"""Инференс одной или нескольких (Seed Ensemble) моделей ранжирования Avito.

Формат выхода:
- Для test: parquet со столбцами feed_id (int32), item_id (int32), score (float32).
- Для val (OOF): дополнительно сохраняется колонка target, и считается итоговый NDCG.

Поддерживает ансамблирование нескольких чекпоинтов:
  python -u src/nn/predict.py --data /data --split test --models m1.pt m2.pt m3.pt --agg rank --out sub.parquet
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.common.metric import all_variants
from src.nn.dataset import NUM_CAND_FEATS, FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel


def normalize_feed_ranks(feed_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Переводит скоры каждой ленты в нормализованные ранги [0..1] (1.0 = лучший)."""
    df = pl.DataFrame({"feed_id": feed_ids, "score": scores})
    ranked = df.with_columns(
        (
            (pl.col("score").rank("ordinal").over("feed_id") - 1.0)
            / pl.max_horizontal(pl.len().over("feed_id") - 1.0, pl.lit(1.0))
        ).cast(pl.Float32).alias("rank")
    )
    return ranked["rank"].to_numpy()


def normalize_feed_zscore(feed_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Нормализует скоры внутри каждой ленты через z-score."""
    df = pl.DataFrame({"feed_id": feed_ids, "score": scores})
    normed = df.with_columns(
        (
            (pl.col("score") - pl.col("score").mean().over("feed_id"))
            / (pl.col("score").std().over("feed_id").fill_null(1.0) + 1e-6)
        ).cast(pl.Float32).alias("z")
    )
    return normed["z"].to_numpy()


@torch.no_grad()
def predict_single_model(
    checkpoint_path: str | Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Прогоняет один чекпоинт через DataLoader, экономя память."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = ckpt.get("epoch", "?")
    val_ndcg = ckpt.get("val_ndcg", ckpt.get("best_ndcg", 0.0))
    print(f"  -> Загружен чекпоинт {Path(checkpoint_path).name} (эпоха {epoch}, Val NDCG: {val_ndcg:.4f})")

    model_config = ckpt.get("model_config", {
        "emb_dim": 128,
        "d_model": 128,
        "nhead": 4,
        "num_categories": 1000,
        "num_cand_feats": NUM_CAND_FEATS,
    })
    model = FeedRankingModel(**model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_feed_ids = []
    all_item_ids = []
    all_scores = []
    all_targets = []
    has_targets = False

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # На инференсе дебиасинг отключен
        _, main_scores = model(batch, use_pos_bias=False)
        cand_mask = batch["candidate_mask"]

        b, m = cand_mask.shape
        feed_ids_exp = batch["feed_id"].unsqueeze(1).expand(b, m)
        item_ids = batch["cand_item_id"]

        mask_np = cand_mask.cpu().numpy()
        all_feed_ids.append(feed_ids_exp.cpu().numpy()[mask_np])
        all_item_ids.append(item_ids.cpu().numpy()[mask_np])
        all_scores.append(main_scores.cpu().numpy()[mask_np])

        if "target" in batch and batch["target"] is not None:
            has_targets = True
            all_targets.append(batch["target"].cpu().numpy()[mask_np])

    # Выгружаем модель из памяти
    del model, ckpt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feed_ids_cat = np.concatenate(all_feed_ids)
    item_ids_cat = np.concatenate(all_item_ids)
    scores_cat = np.concatenate(all_scores)
    targets_cat = np.concatenate(all_targets) if (has_targets and all_targets) else None

    return feed_ids_cat, item_ids_cat, scores_cat, targets_cat


def generate_predictions(
    data_root: str,
    checkpoint_paths: list[str | Path],
    split: str = "test",
    batch_size: int = 32,
    agg: str = "rank",
    weights: list[float] | None = None,
    max_parts: int | None = None,
    max_feeds: int | None = None,
    device_str: str | None = None,
    out_path: str = "submission.parquet",
) -> dict[str, float] | None:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_models = len(checkpoint_paths)

    if weights is not None:
        if len(weights) != n_models:
            raise ValueError(f"Количество весов ({len(weights)}) не совпадает с числом моделей ({n_models})!")
        w_arr = np.asarray(weights, dtype=np.float32)
        w_sum = float(w_arr.sum())
        if w_sum <= 0:
            raise ValueError("Сумма весов должна быть > 0!")
        w_arr = w_arr / w_sum
    else:
        w_arr = np.full(n_models, 1.0 / n_models, dtype=np.float32)

    print("=" * 70)
    print(f"Генерация предсказаний: сплит '{split}' | Моделей: {n_models} | Device: {device}")
    if n_models > 1:
        print(f"Метод агрегации: {agg} | Веса: {[round(float(w), 4) for w in w_arr]}")
    print("=" * 70)

    dataset = FeedRankingDataset(
        data_root=data_root,
        split=split,
        max_parts=max_parts,
        max_feeds=max_feeds,
    )
    print(f"Лент для скоринга: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=0,
    )

    feed_ids = None
    item_ids = None
    targets = None
    model_scores_list = []

    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"[{i + 1}/{len(checkpoint_paths)}] Инференс модели: {ckpt_path}")
        f_ids, i_ids, scores, tgts = predict_single_model(ckpt_path, loader, device)
        if feed_ids is None:
            feed_ids, item_ids, targets = f_ids, i_ids, tgts

        if targets is not None:
            m_single = all_variants(feed_ids, targets, scores)
            print(f"     NDCG одиночной модели: full={m_single['ndcg_full']:.4f} | @30={m_single['ndcg@30']:.4f}")

        model_scores_list.append(scores)

    # Агрегация скоров
    if len(model_scores_list) == 1:
        final_scores = model_scores_list[0].astype(np.float32)
    else:
        if agg == "rank":
            normed = [normalize_feed_ranks(feed_ids, s) for s in model_scores_list]
        elif agg == "zscore":
            normed = [normalize_feed_zscore(feed_ids, s) for s in model_scores_list]
        else:
            normed = model_scores_list

        # Взвешенная сумма с использованием w_arr
        stacked = np.stack(normed, axis=0)  # [K, N]
        final_scores = np.tensordot(w_arr, stacked, axes=([0], [0])).astype(np.float32)

    val_metrics = None
    if targets is not None:
        val_metrics = all_variants(feed_ids, targets, final_scores)
        print("-" * 70)
        print(
            f"[ИТОГОВЫЙ NDCG ({'АНСАМБЛЬ' if len(model_scores_list) > 1 else 'МОДЕЛЬ'})]: "
            f"full={val_metrics['ndcg_full']:.4f} | "
            f"idcg30={val_metrics['ndcg_idcg30']:.4f} | "
            f"@30={val_metrics['ndcg@30']:.4f}"
        )
        print("-" * 70)

    cols = {
        "feed_id": pl.Series(feed_ids, dtype=pl.Int32),
        "item_id": pl.Series(item_ids, dtype=pl.Int32),
        "score": pl.Series(final_scores, dtype=pl.Float32),
    }
    if targets is not None:
        cols["target"] = pl.Series(targets, dtype=pl.Int8)

    df = pl.DataFrame(cols)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)
    print(f"[OK] Файл успешно сохранен: {out_file} (строк: {len(df):,})")

    return val_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Инференс одной модели или Seed Ensemble для Avito")
    parser.add_argument("--data", default="data/synth", help="Корень данных")
    parser.add_argument("--split", default="test", help="Сплит (test или val)")
    parser.add_argument("--model", default=None, help="Путь к одному чекпоинту (обратная совместимость)")
    parser.add_argument("--models", nargs="+", default=None, help="Список путей к чекпоинтам для ансамбля")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Веса моделей для взвешенного усреднения")
    parser.add_argument("--agg", choices=["rank", "zscore", "mean"], default="rank", help="Способ агрегации ансамбля")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-parts", type=int, default=None, help="Ограничить количество партов")
    parser.add_argument("--max-feeds", type=int, default=None, help="Ограничить количество лент")
    parser.add_argument("--out", default="models/nn/run_prod/submission.parquet", help="Путь к выходному parquet (или папке при --save-as-danya)")
    parser.add_argument("--save-as-danya", action="store_true", help="Сохранять как <out_dir>/val_pred.parquet и test_pred.parquet для blend.py/select_final.py Дани")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_path = args.out
    if args.save_as_danya:
        p = Path(args.out)
        out_dir = p.parent if p.suffix == ".parquet" else p
        out_path = str(out_dir / f"{args.split}_pred.parquet")

    if args.models:
        ckpts = args.models
    elif args.model:
        ckpts = [args.model]
    else:
        ckpts = ["models/nn/run_prod/best_model.pt"]

    generate_predictions(
        data_root=args.data,
        checkpoint_paths=ckpts,
        split=args.split,
        batch_size=args.batch_size,
        agg=args.agg,
        weights=args.weights,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds,
        device_str=args.device,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
