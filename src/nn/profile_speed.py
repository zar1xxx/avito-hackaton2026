"""Профайлер скорости и потребления VRAM для PyTorch пайплайна ранжирования лент.

Замеряет:
- Скорость чтения данных (DataLoader time)
- Время передачи на GPU (Host to Device)
- Время Forward и Backward проходов
- Пиковую память GPU (torch.cuda.max_memory_allocated)
- Прогнозируемое время эпохи и пропускную способность (лент в секунду).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.nn.dataset import FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel


def run_profiler(
    data_root: str,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 0,
    num_batches: int = 20,
    device_str: str | None = None,
) -> None:
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 60)
    print(f"Запуск профайлера PyTorch пайплайна")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print("=" * 60)

    dataset = FeedRankingDataset(data_root=data_root, split=split)
    total_feeds = len(dataset)
    print(f"Всего лент в '{split}': {total_feeds}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ranking_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = FeedRankingModel(emb_dim=128, num_categories=1000, num_cand_feats=5, hidden_dim=128)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Синхронизация GPU для точного замера времени
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    t_data_list = []
    t_transfer_list = []
    t_fwd_list = []
    t_bwd_list = []

    print(f"\nСтарт прогона {num_batches} батчей (прогрев 2 батча)...")

    loader_iter = iter(loader)
    warmup = 2
    actual_runs = 0

    t_start_total = time.perf_counter()

    for step in range(num_batches + warmup):
        # 1. Замер загрузки данных
        sync()
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        sync()
        t_data = time.perf_counter() - t0

        # 2. Замер переноса на Device
        sync()
        t1 = time.perf_counter()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        sync()
        t_transfer = time.perf_counter() - t1

        # 3. Замер Forward
        sync()
        t2 = time.perf_counter()
        scores, _ = model(batch)  # [B, M]
        cand_mask = batch["candidate_mask"]

        if "target" in batch and batch["target"] is not None:
            raw_loss = criterion(scores, batch["target"])
            loss = (raw_loss * cand_mask.float()).sum() / torch.clamp(cand_mask.float().sum(), min=1.0)
        else:
            loss = (scores * cand_mask.float()).sum() * 0.0

        sync()
        t_fwd = time.perf_counter() - t2

        # 4. Замер Backward
        sync()
        t3 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sync()
        t_bwd = time.perf_counter() - t3

        if step >= warmup:
            t_data_list.append(t_data)
            t_transfer_list.append(t_transfer)
            t_fwd_list.append(t_fwd)
            t_bwd_list.append(t_bwd)
            actual_runs += 1

    sync()
    t_total = time.perf_counter() - t_start_total

    avg_data = sum(t_data_list) / actual_runs
    avg_trans = sum(t_transfer_list) / actual_runs
    avg_fwd = sum(t_fwd_list) / actual_runs
    avg_bwd = sum(t_bwd_list) / actual_runs
    avg_step = avg_data + avg_trans + avg_fwd + avg_bwd

    feeds_per_sec = (batch_size * actual_runs) / (avg_step * actual_runs)
    epoch_time_min = (total_feeds / feeds_per_sec) / 60.0

    print("\n" + "=" * 60)
    print(" РЕЗУЛЬТАТЫ ПРОФАЙЛИНГА ")
    print("=" * 60)
    print(f"Время шага (среднее):        {avg_step * 1000:.2f} ms")
    print(f"  ├── DataLoader (IO/memmap): {avg_data * 1000:.2f} ms ({avg_data/avg_step*100:.1f}%)")
    print(f"  ├── Transfer to Device:    {avg_trans * 1000:.2f} ms ({avg_trans/avg_step*100:.1f}%)")
    print(f"  ├── Forward pass:          {avg_fwd * 1000:.2f} ms ({avg_fwd/avg_step*100:.1f}%)")
    print(f"  └── Backward + Opt step:   {avg_bwd * 1000:.2f} ms ({avg_bwd/avg_step*100:.1f}%)")
    print("-" * 60)
    print(f"Пропускная способность:      {feeds_per_sec:.1f} лент/сек")
    print(f"Оценка времени 1 эпохи:      {epoch_time_min:.2f} мин ({epoch_time_min*60:.1f} сек) на {total_feeds} лент")

    if device.type == "cuda":
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print("-" * 60)
        print(f"Пиковое потребление VRAM:    {max_vram_mb:.2f} MB (выделено тензорами)")
        print(f"Зарезервировано CUDA памяти: {reserved_vram_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synth", help="Путь к корню данных (содержащему train/)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_profiler(
        data_root=args.data,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        device_str=args.device,
    )
