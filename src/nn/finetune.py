"""Скрипт дообучения (fine-tuning) нейросети на объединенной выборке train + val.

Особенности:
1. Загружает предобученный чекпоинт (лучшую модель первого этапа).
2. Объединяет train и val в единый датасет с повышенным весом для val (val_weight >= 1.0),
   так как val хронологически ближе к test и в val нет позиционного смещения.
3. Обучает модель с меньшим learning rate (по умолчанию 2e-4) для аккуратной адаптации весов.
4. Ведет валидацию по отложенной части val с подсчетом NDCG.
5. Сохраняет итоговый чекпоинт best_model_finetuned.pt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.common.metric import all_variants
from src.nn.dataset import (
    NUM_CAND_FEATS,
    FeedRankingDataset,
    collate_ranking_batch,
)
from src.nn.model import FeedRankingModel


class WeightedCombinedDataset(Dataset):
    """Объединяет два датасета (train и val), назначая сэмплам разные веса для лосса."""

    def __init__(
        self,
        ds_train: Dataset,
        ds_val: Dataset,
        val_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.len_train = len(ds_train)
        self.len_val = len(ds_val)
        self.val_weight = val_weight

    def __len__(self) -> int:
        return self.len_train + self.len_val

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < self.len_train:
            item = self.ds_train[idx]
            item["sample_weight"] = 1.0
            item["is_val"] = False
        else:
            item = self.ds_val[idx - self.len_train]
            item["sample_weight"] = self.val_weight
            item["is_val"] = True
        return item


def weighted_listnet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """Взвешенный ListNet loss: учитывает вес каждой ленты (например, повышенный вес val)."""
    masked_logits = logits.masked_fill(~mask, -1e9)
    pred_log_probs = F.log_softmax(masked_logits, dim=-1)

    scaled_targets = (targets * 3.0) / target_temperature
    masked_targets = scaled_targets.masked_fill(~mask, -1e9)
    true_probs = F.softmax(masked_targets, dim=-1)

    feed_loss = -(true_probs * pred_log_probs).sum(dim=-1)  # [B]

    has_candidates = mask.any(dim=-1)
    if not has_candidates.any():
        return logits.sum() * 0.0

    if weights is not None:
        valid_losses = feed_loss[has_candidates] * weights[has_candidates]
        return valid_losses.sum() / torch.clamp(weights[has_candidates].sum(), min=1e-5)
    else:
        return feed_loss[has_candidates].mean()


@torch.no_grad()
def evaluate_ndcg(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Валидация NDCG по отложенной выборке (без position bias)."""
    model.eval()
    all_feed_ids = []
    all_targets = []
    all_scores = []

    for batch in val_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        _, main_scores = model(batch, use_pos_bias=False)
        mask = batch["candidate_mask"]

        if "target" not in batch or batch["target"] is None:
            continue

        b, m = mask.shape
        feed_ids_exp = batch["feed_id"].unsqueeze(1).expand(b, m)

        val_mask = mask.cpu().numpy()
        feed_ids = feed_ids_exp.cpu().numpy()[val_mask]
        targets = batch["target"].cpu().numpy()[val_mask]
        scores = main_scores.cpu().numpy()[val_mask]

        all_feed_ids.append(feed_ids)
        all_targets.append(targets)
        all_scores.append(scores)

    if not all_feed_ids:
        return {"ndcg_full": 0.0, "ndcg_idcg30": 0.0, "ndcg@30": 0.0}

    return all_variants(np.concatenate(all_feed_ids), np.concatenate(all_targets), np.concatenate(all_scores))


def finetune(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 75)
    print(" ЗАПУСК FINE-TUNING (TRAIN + VAL)")
    print(f"Device: {device} | Base Model: {args.model}")
    print(f"Val Weight: {args.val_weight}x | Fine-tuning LR: {args.lr:.2e}")
    print("=" * 75)

    # 1. Загрузка чекпоинта базовой модели
    ckpt = torch.load(args.model, map_location=device)
    model_config = ckpt.get("model_config", {
        "emb_dim": 128,
        "d_model": 128,
        "nhead": 4,
        "num_categories": 1000,
        "num_cand_feats": NUM_CAND_FEATS,
        "max_hist_len": 256,
        "dropout": 0.15,
    })

    model = FeedRankingModel(**model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Базовая модель успешно загружена (исходный Val NDCG: {ckpt.get('best_ndcg', 0.0):.4f})")

    # 2. Подготовка датасетов
    train_ds = FeedRankingDataset(
        data_root=args.data,
        split=args.train_split,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds,
        max_hist_len=model_config.get("max_hist_len", 256),
    )
    val_ds = FeedRankingDataset(
        data_root=args.data,
        split=args.val_split,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds // 3 if args.max_feeds else None,
        max_hist_len=model_config.get("max_hist_len", 256),
    )

    print(f"Train лент: {len(train_ds):,}, Val лент: {len(val_ds):,}")

    # Объединенный взвешенный датасет для обучения
    combined_ds = WeightedCombinedDataset(train_ds, val_ds, val_weight=args.val_weight)
    print(f"Всего лент в объединенной выборке: {len(combined_ds):,}")

    train_loader = DataLoader(
        combined_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Отдельный загрузчик val для контроля качества
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Оптимизатор и шедулер
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_metrics = evaluate_ndcg(model, val_loader, device)
    print(f"Initial Val NDCG до дообучения: full={init_metrics['ndcg_full']:.4f} | @30={init_metrics['ndcg@30']:.4f}")

    best_ndcg = init_metrics["ndcg_full"]
    history_file = out_dir / "history_finetune.csv"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_ndcg_full,val_ndcg_idcg30,val_ndcg_at30,lr,epoch_time_sec,best_ndcg\n")
        f.write(f"0,NaN,{init_metrics['ndcg_full']:.5f},{init_metrics['ndcg_idcg30']:.5f},{init_metrics['ndcg@30']:.5f},{args.lr:.2e},0.0,{best_ndcg:.5f}\n")

    patience_counter = 0

    # 4. Цикл fine-tuning
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # На train включаем debias, но если в батче лента из val (где pos нет), голова pos не активна
            total_scores, _ = model(batch, use_pos_bias=True)
            cand_mask = batch["candidate_mask"]
            target = batch["target"]
            weights = batch.get("sample_weight")

            loss = weighted_listnet_loss(total_scores, target, cand_mask, weights=weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_duration = time.perf_counter() - epoch_start
        val_metrics = evaluate_ndcg(model, val_loader, device)
        cur_ndcg = val_metrics["ndcg_full"]
        scheduler.step(cur_ndcg)

        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = cur_ndcg > best_ndcg
        if is_best:
            best_ndcg = cur_ndcg
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_data = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "best_ndcg": best_ndcg,
            "val_ndcg": cur_ndcg,
            "metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt_data, out_dir / "last_model_finetuned.pt")

        if is_best:
            torch.save(ckpt_data, out_dir / "best_model_finetuned.pt")

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.5f},{val_metrics['ndcg_full']:.5f},{val_metrics['ndcg_idcg30']:.5f},{val_metrics['ndcg@30']:.5f},{cur_lr:.2e},{epoch_duration:.2f},{best_ndcg:.5f}\n")

        print(
            f"Epoch [{epoch:2d}/{args.epochs}] ({epoch_duration:.2f}с) | "
            f"Loss: {avg_loss:.4f} | "
            f"Val NDCG: full={val_metrics['ndcg_full']:.4f} | @30={val_metrics['ndcg@30']:.4f} | "
            f"lr: {cur_lr:.2e} "
            f"{'(* NEW BEST *)' if is_best else f'(patience: {patience_counter}/{args.patience})'}"
        )

        if patience_counter >= args.patience:
            print(f"\n[EarlyStopping] Fine-tuning остановлен по patience={args.patience}")
            break

    print(f"\n[DONE] Fine-tuning завершен. Лучший Val NDCG: {best_ndcg:.4f}")
    print(f"Финальный чекпоинт сохранен: {out_dir / 'best_model_finetuned.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning нейросетевого ранжирования Avito")
    parser.add_argument("--data", default="data/synth", help="Путь к корню данных")
    parser.add_argument("--model", default="models/nn/run_prod/best_model.pt", help="Путь к базовому чекпоинту")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--out", default="models/nn/finetuned", help="Папка для сохранения")
    parser.add_argument("--val-weight", type=float, default=2.0, help="Весовой коэффициент для лент из val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="Пониженный LR для finetune")
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--max-feeds", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    finetune(args)


if __name__ == "__main__":
    main()
