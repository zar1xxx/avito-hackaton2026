"""Скрипт обучения продвинутой нейросетевой модели ранжирования рекомендаций Avito.

Включает:
- History Transformer Encoder + Candidate-to-History Cross-Attention
- Positional Debias: обучение с головой позиции на train, отключение на val/test
- ListNet ranking loss с температурой
- Оценку валидационного NDCG по чистым контентным скорам
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.metric import all_variants
from src.nn.dataset import FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel


def listnet_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet ranking loss с маскированием паддингов."""
    masked_logits = logits.masked_fill(~mask, -1e9)
    pred_log_probs = F.log_softmax(masked_logits, dim=-1)

    # Нормализуем таргеты внутри ленты: позитивы получают больший вес
    scaled_targets = (targets * 3.0) / target_temperature
    masked_targets = scaled_targets.masked_fill(~mask, -1e9)
    true_probs = F.softmax(masked_targets, dim=-1)

    # Cross-entropy по лентам
    feed_loss = -(true_probs * pred_log_probs).sum(dim=-1)

    has_candidates = mask.any(dim=-1)
    if has_candidates.any():
        return feed_loss[has_candidates].mean()
    return logits.sum() * 0.0


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary Cross Entropy с маскированием паддингов."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mask_f = mask.float()
    return (bce * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)


@torch.no_grad()
def evaluate_ndcg(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Считает метрики NDCG по валидационному датасету с отключенным position bias."""
    model.eval()
    all_feed_ids = []
    all_targets = []
    all_scores = []

    for batch in val_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Валидация ВСЕГДА без position bias (чистый main_score)
        _, main_scores = model(batch, use_pos_bias=False)
        mask = batch["candidate_mask"]

        if "target" not in batch or batch["target"] is None:
            continue

        b, m = mask.shape
        feed_ids_expanded = batch["feed_id"].unsqueeze(1).expand(b, m)

        val_cand_mask = mask.cpu().numpy()
        feed_ids = feed_ids_expanded.cpu().numpy()[val_cand_mask]
        targets = batch["target"].cpu().numpy()[val_cand_mask]
        scores = main_scores.cpu().numpy()[val_cand_mask]

        all_feed_ids.append(feed_ids)
        all_targets.append(targets)
        all_scores.append(scores)

    if not all_feed_ids:
        return {"ndcg_full": 0.0, "ndcg_idcg30": 0.0, "ndcg@30": 0.0}

    feed_ids_cat = np.concatenate(all_feed_ids)
    targets_cat = np.concatenate(all_targets)
    scores_cat = np.concatenate(all_scores)

    return all_variants(feed_ids_cat, targets_cat, scores_cat)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 70)
    print(f"Запуск обучения: {args.data} | Device: {device} | Loss: {args.loss}")
    print(f"Positional Debias: {'ВКЛЮЧЕН (обучение головы pos)' if args.use_pos_bias else 'ОТКЛЮЧЕН'}")
    print("=" * 70)

    # 1. Загрузка данных
    train_dataset = FeedRankingDataset(
        data_root=args.data,
        split="train",
        max_hist_len=args.max_hist_len,
    )
    val_dataset = FeedRankingDataset(
        data_root=args.data,
        split="val",
        max_hist_len=args.max_hist_len,
    )

    print(f"Train лент: {len(train_dataset)}, Val лент: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ranking_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ranking_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Инициализация продвинутой модели
    model = FeedRankingModel(
        emb_dim=128,
        d_model=args.hidden_dim,
        nhead=args.nhead,
        num_categories=args.num_categories,
        num_cand_feats=5,
        max_hist_len=args.max_hist_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров модели: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Начальная оценка до обучения (Zero-shot)
    init_metrics = evaluate_ndcg(model, val_loader, device)
    print(f"Initial Val NDCG (без обучения): full={init_metrics['ndcg_full']:.4f} | idcg30={init_metrics['ndcg_idcg30']:.4f} | @30={init_metrics['ndcg@30']:.4f}")

    best_ndcg = -1.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Цикл обучения
    global_step = 0
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            # На train включаем position bias, если доступно
            total_scores, _ = model(batch, use_pos_bias=args.use_pos_bias)
            cand_mask = batch["candidate_mask"]
            target = batch["target"]

            if args.loss == "listnet":
                loss = listnet_loss(total_scores, target, cand_mask)
            else:
                loss = masked_bce_loss(total_scores, target, cand_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1

            if global_step % args.log_every == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch}/{args.epochs}] | Step {batch_idx + 1:3d}/{len(train_loader)} | "
                    f"Train Loss: {loss_val:.4f} (avg: {np.mean(epoch_losses):.4f})"
                )

        epoch_duration = time.perf_counter() - epoch_start
        train_throughput = len(train_dataset) / max(epoch_duration, 1e-6)

        # Валидация после эпохи (чистый контентный скор)
        val_metrics = evaluate_ndcg(model, val_loader, device)
        cur_ndcg = val_metrics["ndcg_full"]

        is_best = cur_ndcg > best_ndcg
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_ndcg": cur_ndcg,
            "metrics": val_metrics,
        }
        torch.save(ckpt_data, out_dir / "last_model.pt")

        if is_best:
            best_ndcg = cur_ndcg
            torch.save(ckpt_data, out_dir / "best_model.pt")

        print("-" * 70)
        print(
            f"--> Конец Эпохи {epoch} ({epoch_duration:.2f}с, {train_throughput:.1f} лент/сек) | "
            f"Val NDCG (unbiased): full={val_metrics['ndcg_full']:.4f} | "
            f"idcg30={val_metrics['ndcg_idcg30']:.4f} | "
            f"@30={val_metrics['ndcg@30']:.4f} "
            f"{'(* NEW BEST *)' if is_best else ''}"
        )
        print("-" * 70)

    total_time = time.perf_counter() - total_start
    print(f"\n[DONE] Обучение завершено за {total_time:.2f}с. Лучший Val NDCG(full): {best_ndcg:.4f}")
    print(f"Чекпоинт сохранен в: {out_dir / 'best_model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение продвинутой нейросети ранжирования Avito")
    parser.add_argument("--data", default="data/synth", help="Путь к корню данных (train/val)")
    parser.add_argument("--out", default="models/nn/transformer_debias", help="Папка чекпоинта")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-categories", type=int, default=1000)
    parser.add_argument("--max-hist-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", choices=["listnet", "bce"], default="listnet")
    parser.add_argument("--use-pos-bias", action="store_true", default=True, help="Включить голову debias для pos на train")
    parser.add_argument("--no-pos-bias", dest="use_pos_bias", action="store_false")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
