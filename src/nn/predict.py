"""Инференс модели ранжирования лент Avito и генерация сабмита.

Формат выхода:
parquet со столбцами:
feed_id: int32
item_id: int32
score: float32

Сверяется со скриптом scripts/validate_submission.py.
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
import torch
from torch.utils.data import DataLoader

from src.nn.dataset import FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel


@torch.no_grad()
def generate_predictions(
    data_root: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 32,
    device_str: str | None = None,
    out_path: str = "submission.parquet",
) -> None:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Генерация предсказаний для сплита '{split}' | Device: {device}")

    # Загружаем чекпоинт
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Загружен чекпоинт из эпохи {ckpt.get('epoch', '?')} (Best Val NDCG: {ckpt.get('best_ndcg', 0.0):.4f})")

    # Инициализация датасета и загрузчика
    dataset = FeedRankingDataset(data_root=data_root, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=0,
    )

    # Инициализация модели
    model = FeedRankingModel(emb_dim=128, d_model=128, nhead=4, num_categories=1000).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_feed_ids = []
    all_item_ids = []
    all_scores = []

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Инференс ВСЕГДА с use_pos_bias=False
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

    df = pl.DataFrame({
        "feed_id": pl.Series(np.concatenate(all_feed_ids), dtype=pl.Int32),
        "item_id": pl.Series(np.concatenate(all_item_ids), dtype=pl.Int32),
        "score": pl.Series(np.concatenate(all_scores), dtype=pl.Float32),
    })

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)
    print(f"Сабмит успешно сохранен: {out_file} (строк: {len(df):,})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="models/nn/transformer_debias/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="models/nn/transformer_debias/submission.parquet")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    generate_predictions(
        data_root=args.data,
        checkpoint_path=args.model,
        split=args.split,
        batch_size=args.batch_size,
        device_str=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
