"""Экспорт нейросетевых OOF-признаков для GBDT (CatBoost/LightGBM) по контракту Дани.

Контракт (см. src/gbdt/extra_features.py в ветке danya):
  - Путь: <out>/avgust/{train,val,test}.parquet
  - Ключи: feed_id: int32, item_id: int32 (без дублей)
  - Признаки (все с префиксом x_avgust_, float32):
      * x_avgust_nn_score  — чистый контентный скор нейросети (без position bias)
      * x_avgust_nn_rank   — нормализованный ранг кандидата внутри ленты [0..1]
      * x_avgust_nn_z      — z-score внутри ленты
      * x_avgust_dot_user  — косинус кандидата и пользователя в пространстве NN
      * x_avgust_dot_hist  — косинус кандидата и контекста истории
      * x_avgust_dot_uh    — косинус пользователя и контекста истории

Анти-утечка:
  - Для сплита train признаки считаются строго в режиме K-Fold Out-Of-Fold (OOF):
    модель, предсказывающая признаки для ленты, никогда не видела её при обучении.
  - Для сплитов val и test предсказания усредняются по всем K фолдам.

Пример запуска:
  python -u src/nn/export_gbdt_features.py --data data/synth --folds 3 --epochs 2 --out features
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.common.metric import all_variants
from src.nn.dataset import NUM_CAND_FEATS, FeedRankingDataset, collate_ranking_batch
from src.nn.model import FeedRankingModel
from src.nn.train import apply_in_feed_mixup, evaluate_ndcg, listnet_loss, seed_everything


@torch.no_grad()
def extract_features_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Извлекает предсказания и внутренние сходства нейросети по всем батчам."""
    model.eval()
    feed_ids_list = []
    item_ids_list = []
    scores_list = []
    dot_u_list = []
    dot_h_list = []
    dot_uh_list = []

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        _, main_scores, feats = model(batch, use_pos_bias=False, return_features=True)
        cand_mask = batch["candidate_mask"]

        b, m = cand_mask.shape
        feed_ids_exp = batch["feed_id"].unsqueeze(1).expand(b, m)
        item_ids = batch["cand_item_id"]

        mask_np = cand_mask.cpu().numpy()
        feed_ids_list.append(feed_ids_exp.cpu().numpy()[mask_np])
        item_ids_list.append(item_ids.cpu().numpy()[mask_np])
        scores_list.append(main_scores.cpu().numpy()[mask_np])
        dot_u_list.append(feats["dot_user"].cpu().numpy()[mask_np])
        dot_h_list.append(feats["dot_hist"].cpu().numpy()[mask_np])
        dot_uh_list.append(feats["dot_uh"].cpu().numpy()[mask_np])

    return {
        "feed_id": np.concatenate(feed_ids_list).astype(np.int32),
        "item_id": np.concatenate(item_ids_list).astype(np.int32),
        "score": np.concatenate(scores_list).astype(np.float32),
        "dot_user": np.concatenate(dot_u_list).astype(np.float32),
        "dot_hist": np.concatenate(dot_h_list).astype(np.float32),
        "dot_uh": np.concatenate(dot_uh_list).astype(np.float32),
    }


def build_feature_dataframe(raw_dict: dict[str, np.ndarray]) -> pl.DataFrame:
    """Строит Polars DataFrame с внутрилентовыми рангами, z-скорами и префиксами x_avgust_*."""
    df = pl.DataFrame({
        "feed_id": pl.Series(raw_dict["feed_id"], dtype=pl.Int32),
        "item_id": pl.Series(raw_dict["item_id"], dtype=pl.Int32),
        "x_avgust_nn_score": pl.Series(raw_dict["score"], dtype=pl.Float32),
        "x_avgust_dot_user": pl.Series(raw_dict["dot_user"], dtype=pl.Float32),
        "x_avgust_dot_hist": pl.Series(raw_dict["dot_hist"], dtype=pl.Float32),
        "x_avgust_dot_uh": pl.Series(raw_dict["dot_uh"], dtype=pl.Float32),
    })

    # Дедупликация по (feed_id, item_id) по контракту extra_features.py Дани
    n_before = len(df)
    df = df.unique(subset=["feed_id", "item_id"], keep="first", maintain_order=True)
    if len(df) != n_before:
        print(f"  [INFO] Удалено дублей (feed_id, item_id): {n_before - len(df)}")

    # Внутрилентовые статистики по x_avgust_nn_score
    df = df.with_columns([
        (
            (pl.col("x_avgust_nn_score").rank("ordinal").over("feed_id") - 1.0)
            / pl.max_horizontal(pl.len().over("feed_id") - 1.0, pl.lit(1.0))
        ).cast(pl.Float32).alias("x_avgust_nn_rank"),
        (
            (pl.col("x_avgust_nn_score") - pl.col("x_avgust_nn_score").mean().over("feed_id"))
            / (pl.col("x_avgust_nn_score").std().over("feed_id").fill_null(1.0) + 1e-6)
        ).fill_nan(0.0).cast(pl.Float32).alias("x_avgust_nn_z"),
    ])

    # Упорядочиваем колонки по контракту
    cols = [
        "feed_id",
        "item_id",
        "x_avgust_nn_score",
        "x_avgust_nn_rank",
        "x_avgust_nn_z",
        "x_avgust_dot_user",
        "x_avgust_dot_hist",
        "x_avgust_dot_uh",
    ]
    return df.select(cols)


def train_fold_model(
    train_subset: Subset,
    val_loader: DataLoader,
    args: argparse.Namespace,
    fold_idx: int,
    device: torch.device,
) -> FeedRankingModel:
    """Обучает модель одного фолда с выбором лучшего чекпоинта по val NDCG."""
    seed_everything(args.seed + fold_idx)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed + fold_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=dl_gen,
        collate_fn=collate_ranking_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = FeedRankingModel(
        emb_dim=128,
        d_model=args.hidden_dim,
        nhead=args.nhead,
        num_categories=args.num_categories,
        num_cand_feats=NUM_CAND_FEATS,
        max_hist_len=args.max_hist_len,
        dropout=args.dropout,
        use_set_attention=getattr(args, "use_set_attention", True),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    best_ndcg = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            if args.mixup_alpha > 0.0:
                batch = apply_in_feed_mixup(batch, args.mixup_alpha)

            total_scores, _ = model(batch, use_pos_bias=args.use_pos_bias)
            loss = listnet_loss(
                total_scores,
                batch["target"],
                batch["candidate_mask"],
                logit_temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate_ndcg(model, val_loader, device)
        cur_ndcg = val_metrics["ndcg_full"]
        scheduler.step(cur_ndcg)

        is_best = cur_ndcg > best_ndcg
        if is_best:
            best_ndcg = cur_ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  [Fold {fold_idx + 1}/{args.folds}] Ep {epoch:2d}/{args.epochs} | "
            f"Loss={np.mean(losses):.4f} | Val NDCG={cur_ndcg:.4f} {'(*)' if is_best else ''}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def export_features(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_group_dir = Path(args.out) / "avgust"
    out_group_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f" ЭКСПОРТ НЕЙРО-ФИЧЕЙ ДЛЯ GBDT (КОНТРАКТ: {out_group_dir})")
    print(f" Folds={args.folds} | Epochs={args.epochs} | Device={device}")
    print("=" * 75)

    t0 = time.perf_counter()

    # 1. Загружаем датасеты
    train_ds = FeedRankingDataset(
        data_root=args.data,
        split="train",
        max_parts=args.max_parts,
        max_feeds=args.max_feeds,
        max_hist_len=args.max_hist_len,
    )
    val_ds = FeedRankingDataset(
        data_root=args.data,
        split="val",
        max_parts=args.max_parts,
        max_feeds=args.max_feeds // 3 if args.max_feeds else None,
        max_hist_len=args.max_hist_len,
    )

    has_test = True
    try:
        test_ds = FeedRankingDataset(
            data_root=args.data,
            split="test",
            max_parts=args.max_parts,
            max_feeds=args.max_feeds,
            max_hist_len=args.max_hist_len,
        )
    except FileNotFoundError:
        has_test = False
        test_ds = None

    print(f"Лент: train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds) if has_test else 0:,}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ranking_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ranking_batch) if has_test else None

    # 2. Разбиваем индексы train на K фолдов (каждый индекс = 1 лента feed_id)
    n_train = len(train_ds)
    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(n_train)
    folds_indices = np.array_split(all_indices, args.folds)

    # Накопители для OOF на train и усреднения на val/test
    oof_train_parts: list[dict[str, np.ndarray]] = []
    val_accum: dict[str, np.ndarray] | None = None
    test_accum: dict[str, np.ndarray] | None = None

    for k in range(args.folds):
        print(f"\n>>> Обучение фолда {k + 1}/{args.folds}")
        oof_idx = folds_indices[k]
        tr_idx = np.concatenate([folds_indices[j] for j in range(args.folds) if j != k])

        train_subset = Subset(train_ds, tr_idx.tolist())
        oof_subset = Subset(train_ds, oof_idx.tolist())
        oof_loader = DataLoader(oof_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ranking_batch)

        model = train_fold_model(train_subset, val_loader, args, k, device)

        # Предсказание OOF на невиданной части train
        oof_res = extract_features_loader(model, oof_loader, device)
        oof_train_parts.append(oof_res)

        # Предсказания на val и test (будут усреднены по K фолдам)
        val_res = extract_features_loader(model, val_loader, device)
        if val_accum is None:
            val_accum = {key: val.copy() for key, val in val_res.items()}
        else:
            for feat_key in ("score", "dot_user", "dot_hist", "dot_uh"):
                val_accum[feat_key] += val_res[feat_key]

        if has_test and test_loader is not None:
            test_res = extract_features_loader(model, test_loader, device)
            if test_accum is None:
                test_accum = {key: val.copy() for key, val in test_res.items()}
            else:
                for feat_key in ("score", "dot_user", "dot_hist", "dot_uh"):
                    test_accum[feat_key] += test_res[feat_key]

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3. Сохранение train.parquet (строго OOF!)
    train_raw = {
        key: np.concatenate([p[key] for p in oof_train_parts])
        for key in ("feed_id", "item_id", "score", "dot_user", "dot_hist", "dot_uh")
    }
    train_df = build_feature_dataframe(train_raw).sort(["feed_id", "item_id"])
    train_path = out_group_dir / "train.parquet"
    train_df.write_parquet(train_path)
    print(f"\n[OK] Сохранен OOF train: {train_path} (строк: {len(train_df):,})")

    # 4. Усреднение и сохранение val.parquet
    assert val_accum is not None
    for feat_key in ("score", "dot_user", "dot_hist", "dot_uh"):
        val_accum[feat_key] /= float(args.folds)
    val_df = build_feature_dataframe(val_accum).sort(["feed_id", "item_id"])
    val_path = out_group_dir / "val.parquet"
    val_df.write_parquet(val_path)
    print(f"[OK] Сохранен val:       {val_path} (строк: {len(val_df):,})")

    # Проверка итогового ансамблевого NDCG по экспортированному скору на val
    val_targets = []
    for batch in val_loader:
        mask_np = batch["candidate_mask"].numpy()
        b, m = mask_np.shape
        f_exp = batch["feed_id"].unsqueeze(1).expand(b, m).numpy()[mask_np]
        i_exp = batch["cand_item_id"].numpy()[mask_np]
        t_exp = batch["target"].numpy()[mask_np]
        val_targets.append(pl.DataFrame({"feed_id": f_exp, "item_id": i_exp, "target": t_exp}))
    val_t_df = pl.concat(val_targets).cast({"feed_id": pl.Int32, "item_id": pl.Int32})
    val_joined = val_df.join(val_t_df, on=["feed_id", "item_id"], how="inner")
    m_val = all_variants(
        val_joined["feed_id"].to_numpy(),
        val_joined["target"].to_numpy(),
        val_joined["x_avgust_nn_score"].to_numpy(),
    )
    print(f"     Качество экспортируемого x_avgust_nn_score на val: full={m_val['ndcg_full']:.4f} | @30={m_val['ndcg@30']:.4f}")

    # 5. Усреднение и сохранение test.parquet
    if has_test and test_accum is not None:
        for feat_key in ("score", "dot_user", "dot_hist", "dot_uh"):
            test_accum[feat_key] /= float(args.folds)
        test_df = build_feature_dataframe(test_accum).sort(["feed_id", "item_id"])
        test_path = out_group_dir / "test.parquet"
        test_df.write_parquet(test_path)
        print(f"[OK] Сохранен test:      {test_path} (строк: {len(test_df):,})")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 75)
    print(f" Экспорт нейро-фичей завершен за {elapsed:.1f}с!")
    print(f" Подключение к GBDT Дани: --features-dir {args.out} --extra avgust")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт K-Fold OOF нейро-фичей для GBDT (контракт Дани)")
    parser.add_argument("--data", required=True, help="Путь к корню данных")
    parser.add_argument("--out", default="features", help="Корневая папка признаков (создаст <out>/avgust/)")
    parser.add_argument("--folds", type=int, default=4, help="Число фолдов для честного OOF на train")
    parser.add_argument("--epochs", type=int, default=6, help="Эпох обучения каждого фолда")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-categories", type=int, default=1000)
    parser.add_argument("--max-hist-len", type=int, default=256)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-pos-bias", action="store_true", default=True)
    parser.add_argument("--no-pos-bias", dest="use_pos_bias", action="store_false")
    parser.add_argument("--use_set_attention", "--use-set-attention", dest="use_set_attention", action="store_true", default=True)
    parser.add_argument("--no_set_attention", "--no-set-attention", dest="use_set_attention", action="store_false")
    parser.add_argument("--max-parts", type=int, default=None)
    parser.add_argument("--max-feeds", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    export_features(args)


if __name__ == "__main__":
    main()
