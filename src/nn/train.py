"""Скрипт обучения продвинутой нейросетевой модели ранжирования рекомендаций Avito.

Включает:
- History Transformer Encoder + Candidate-to-History Cross-Attention
- Positional Debias: обучение с головой позиции на train, отключение на val/test
- ListNet ranking loss с температурой
- EarlyStopping и ReduceLROnPlateau scheduler
- Оценку валидационного NDCG по чистым контентным скорам
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.metric import all_variants
from src.nn.dataset import NUM_CAND_FEATS, FeedRankingDataset, collate_ranking_batch
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

    scaled_targets = (targets * 3.0) / target_temperature
    masked_targets = scaled_targets.masked_fill(~mask, -1e9)
    true_probs = F.softmax(masked_targets, dim=-1)

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
    print("=" * 75)
    print(f"Запуск обучения: {args.data} | Device: {device} | Loss: {args.loss}")
    print(f"Сплиты: train='{args.train_split}', val='{args.val_split}'")
    print(f"Debias: {'ВКЛЮЧЕН' if args.use_pos_bias else 'ОТКЛЮЧЕН'} | EarlyStopping patience: {args.patience}")
    print("=" * 75)

    # 1. Загрузка данных
    train_dataset = FeedRankingDataset(
        data_root=args.data,
        split=args.train_split,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds,
        max_hist_len=args.max_hist_len,
    )
    val_dataset = FeedRankingDataset(
        data_root=args.data,
        split=args.val_split,
        max_parts=args.max_parts,
        max_feeds=args.max_feeds // 3 if args.max_feeds else None,
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

    # Конфиг модели для сохранения в чекпоинт
    model_config = {
        "emb_dim": 128,
        "d_model": args.hidden_dim,
        "nhead": args.nhead,
        "num_categories": args.num_categories,
        "num_cand_feats": NUM_CAND_FEATS,
        "max_hist_len": args.max_hist_len,
        "dropout": args.dropout,
    }

    # 2. Инициализация продвинутой модели
    model = FeedRankingModel(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров модели: {total_params:,} (входных числовых фичей: {NUM_CAND_FEATS})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    # 3. Начальная оценка до обучения
    init_metrics = evaluate_ndcg(model, val_loader, device)
    print(
        f"Initial Val NDCG (без обучения): full={init_metrics['ndcg_full']:.4f} | "
        f"idcg30={init_metrics['ndcg_idcg30']:.4f} | @30={init_metrics['ndcg@30']:.4f}"
    )

    best_ndcg = -1.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация файла истории метрик
    history_file = out_dir / "history.csv"
    with open(history_file, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_ndcg_full,val_ndcg_idcg30,val_ndcg_at30,lr,epoch_time_sec,best_ndcg\n")
        f.write(f"0,NaN,{init_metrics['ndcg_full']:.5f},{init_metrics['ndcg_idcg30']:.5f},{init_metrics['ndcg@30']:.5f},{args.lr:.2e},0.0,{init_metrics['ndcg_full']:.5f}\n")

    # 4. Цикл обучения с EarlyStopping
    global_step = 0
    patience_counter = 0
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

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
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:2d}/{args.epochs}] | Step {batch_idx + 1:3d}/{len(train_loader)} | "
                    f"Train Loss: {loss_val:.4f} (avg: {np.mean(epoch_losses):.4f}) | lr: {cur_lr:.2e}"
                )

        epoch_duration = time.perf_counter() - epoch_start
        train_throughput = len(train_dataset) / max(epoch_duration, 1e-6)

        # Валидация после эпохи
        val_metrics = evaluate_ndcg(model, val_loader, device)
        cur_ndcg = val_metrics["ndcg_full"]

        # Шаг шедулера по валидационному NDCG
        scheduler.step(cur_ndcg)
        cur_lr = optimizer.param_groups[0]["lr"]

        is_best = cur_ndcg > best_ndcg
        ckpt_data = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_ndcg": best_ndcg,
            "val_ndcg": cur_ndcg,
            "metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(ckpt_data, out_dir / "last_model.pt")

        if is_best:
            best_ndcg = cur_ndcg
            patience_counter = 0
            torch.save(ckpt_data, out_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Запись эпохи в history.csv
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{avg_train_loss:.5f},{val_metrics['ndcg_full']:.5f},"
                f"{val_metrics['ndcg_idcg30']:.5f},{val_metrics['ndcg@30']:.5f},"
                f"{cur_lr:.2e},{epoch_duration:.2f},{best_ndcg:.5f}\n"
            )

        print("-" * 75)
        print(
            f"--> Конец Эпохи {epoch} ({epoch_duration:.2f}с, {train_throughput:.1f} лент/сек) | "
            f"Val NDCG: full={val_metrics['ndcg_full']:.4f} | "
            f"idcg30={val_metrics['ndcg_idcg30']:.4f} | "
            f"@30={val_metrics['ndcg@30']:.4f} "
            f"{'(* NEW BEST *)' if is_best else f'(patience: {patience_counter}/{args.patience})'}"
        )
        print("-" * 75)

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n[EarlyStopping] NDCG не улучшался {args.patience} эпох. Остановка обучения!")
            break

    total_time = time.perf_counter() - total_start
    print(f"\n[DONE] Обучение завершено за {total_time:.2f}с. Лучший Val NDCG(full): {best_ndcg:.4f}")
    print(f"Чекпоинт сохранен в: {out_dir / 'best_model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение продвинутой нейросети ранжирования Avito")
    parser.add_argument("--data", default="data/synth", help="Путь к корню данных")
    parser.add_argument("--train-split", default="train", help="Имя сплита для train")
    parser.add_argument("--val-split", default="val", help="Имя сплита для val")
    parser.add_argument("--out", default="models/nn/run_prod", help="Папка чекпоинта")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=2, help="Patience для ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-categories", type=int, default=1000)
    parser.add_argument("--max-hist-len", type=int, default=256)
    parser.add_argument("--max-parts", type=int, default=None, help="Ограничить количество читаемых партов (для быстрого дебага)")
    parser.add_argument("--max-feeds", type=int, default=None, help="Ограничить общее количество читаемых лент (защита от OOM)")
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
