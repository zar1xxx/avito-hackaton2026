"""Скрипт проверки FeedRankingDataset и collate_ranking_batch на синтетических данных.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synth", help="Путь к синтетическому датасету")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-hist-len", type=int, default=256)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from src.nn.dataset import FeedRankingDataset, collate_ranking_batch

    print(f"=== Проверка FeedRankingDataset на {args.data} ===")
    dataset = FeedRankingDataset(
        data_root=args.data,
        split="train",
        max_hist_len=args.max_hist_len,
    )

    print(f"Всего лент в train: {len(dataset)}")
    first_sample = dataset[0]
    print("\n--- Пример одной ленты (sample 0) ---")
    print(f"feed_id: {first_sample['feed_id']}, user_id: {first_sample['user_id']}")
    print(f"user_emb shape: {first_sample['user_emb'].shape}, dtype: {first_sample['user_emb'].dtype}")
    print(f"cand_len: {first_sample['cand_len']}, cand_emb shape: {first_sample['cand_emb'].shape}")
    print(f"cand_has_emb: {first_sample['cand_has_emb'].sum()}/{first_sample['cand_len']} имеют эмбеддинг")
    print(f"hist_len: {first_sample['hist_len']}, hist_emb shape: {first_sample['hist_emb'].shape}")
    print(f"hist_has_emb: {first_sample['hist_has_emb'].sum()}/{first_sample['hist_len']} имеют эмбеддинг")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
    )

    batch = next(iter(loader))
    print(f"\n--- Собранный батч (batch_size={args.batch_size}) ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: shape={str(tuple(v.shape)):16s} dtype={v.dtype}")

    # Валидация масок и инвариантов
    b = len(batch["feed_id"])
    m_max = batch["candidate_mask"].shape[1]
    h_max = batch["history_mask"].shape[1]

    # 1. Проверка маски кандидатов
    cand_counts = batch["candidate_mask"].sum(dim=1)
    assert torch.equal(cand_counts, batch["cand_len"]), "Несовпадение candidate_mask с cand_len!"

    # 2. Проверка маски истории
    hist_counts = batch["history_mask"].sum(dim=1)
    assert torch.equal(hist_counts, batch["hist_len"]), "Несовпадение history_mask с hist_len!"

    # 3. Эмбеддинги на позициях паддинга строго нули
    pad_mask = ~batch["candidate_mask"]
    pad_embs = batch["cand_emb"][pad_mask]
    assert torch.all(pad_embs == 0.0), "Эмбеддинги кандидатов на паддингах должны быть нулями!"

    # 4. Проверка has_emb_mask
    assert torch.all(batch["cand_has_emb_mask"] <= batch["candidate_mask"]), (
        "cand_has_emb_mask не может быть True на паддингах!"
    )
    assert torch.all(batch["hist_has_emb_mask"] <= batch["history_mask"]), (
        "hist_has_emb_mask не может быть True на паддингах истории!"
    )

    print("\n[OK] Все ассерты пройдены успешно!")
    print(f"  - M_max в текущем батче: {m_max}")
    print(f"  - H_max в текущем батче: {h_max}")
    print(f"  - Кандидатов всего: {cand_counts.tolist()}")
    print(f"  - Событий истории всего: {hist_counts.tolist()}")


if __name__ == "__main__":
    main()
