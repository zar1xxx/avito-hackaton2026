"""Инференс модели ранжирования лент Avito и генерация сабмита / OOF-предсказаний.

Формат выхода:
- Для test: parquet со столбцами feed_id (int32), item_id (int32), score (float32).
- Для val (OOF): дополнительно сохраняется колонка target для последующего блендинга с GBDT.

Полностью совместим с scripts/validate_submission.py.
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
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import NUM_CAND_FEATS, FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel


@torch.no_grad()
def generate_predictions(
    data_root: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 32,
    max_parts: int | None = None,
    max_feeds: int | None = None,
    device_str: str | None = None,
    out_path: str = "submission.parquet",
) -> None:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 65)
    print(f"Генерация предсказаний: сплит '{split}' | Device: {device}")
    print(f"Чекпоинт: {checkpoint_path}")
    print("=" * 65)

    # 1. Загрузка чекпоинта
    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = ckpt.get("epoch", "?")
    best_ndcg = ckpt.get("best_ndcg", 0.0)
    print(f"Загружен чекпоинт из эпохи {epoch} (Val NDCG: {best_ndcg:.4f})")

    # 2. Инициализация датасета
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

    # 3. Восстановление модели
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

        # Инференс ВСЕГДА с отключенным position bias
        _, main_scores = model(batch, use_pos_bias=False)
        cand_mask = batch["candidate_mask"]

        b, m = cand_mask.shape
        feed_ids_exp = batch["feed_id"].unsqueeze(1).expand(b, m)
        item_ids = batch["cand_item_id"]

        mask_np = cand_mask.cpu().numpy()
        feed_ids_np = feed_ids_exp.cpu().numpy()[mask_np]
        item_ids_np = item_ids.cpu().numpy()[mask_np]
        scores_np = main_scores.cpu().numpy()[mask_np]

        all_feed_ids.append(feed_ids_np)
        all_item_ids.append(item_ids_np)
        all_scores.append(scores_np)

        if "target" in batch and batch["target"] is not None:
            has_targets = True
            all_targets.append(batch["target"].cpu().numpy()[mask_np])

    cols = {
        "feed_id": pl.Series(np.concatenate(all_feed_ids), dtype=pl.Int32),
        "item_id": pl.Series(np.concatenate(all_item_ids), dtype=pl.Int32),
        "score": pl.Series(np.concatenate(all_scores), dtype=pl.Float32),
    }

    if has_targets and all_targets:
        cols["target"] = pl.Series(np.concatenate(all_targets), dtype=pl.Int8)

    df = pl.DataFrame(cols)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)
    print(f"Файл успешно сохранен: {out_file} (строк: {len(df):,})")
    if "target" in df.columns:
        print("Колонка 'target' включена (OOF-формат для ансамбля)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Инференс модели ранжирования лент Avito")
    parser.add_argument("--data", default="data/synth", help="Корень данных")
    parser.add_argument("--split", default="test", help="Сплит для инференса (test или val)")
    parser.add_argument("--model", default="models/nn/run_prod/best_model.pt", help="Путь к чекпоинту")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-parts", type=int, default=None, help="Ограничить количество партов")
    parser.add_argument("--max-feeds", type=int, default=None, help="Ограничить количество лент")
    parser.add_argument("--out", default="models/nn/run_prod/submission.parquet", help="Путь к выходному parquet")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    generate_predictions(
        data_root=args.data,
        checkpoint_path=args.model,
        split=args.split,
        batch_size=args.batch_size,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds,
        device_str=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
