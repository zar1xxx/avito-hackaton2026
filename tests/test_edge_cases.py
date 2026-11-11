"""Unit-тесты для проверки устойчивости модели и датасета к краевым случаям (edge cases).

Проверяет:
1. Пользователь без истории (H=0). Отсутствие NaN / Inf в скорах и градиентах.
2. Все кандидаты в ленте без эмбеддингов (item_id__embedding_idx = -1).
3. Лента с ровно одним кандидатом (M=1).
4. Максимальная длина истории (H=256).
5. Разные размеры лент в одном батче (динамический паддинг и маски).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import pytest
import torch

from src.nn.dataset import (
    CAND_FEAT_NAMES,
    EMB_DIM,
    NUM_CAND_FEATS,
    FeedRankingDataset,
    ShardedFeedRankingDataset,
    collate_ranking_batch,
    compute_candidate_features,
)
from src.nn.model import FeedRankingModel


def make_dummy_sample(
    feed_id: int = 1,
    user_id: int = 100,
    m: int = 10,
    h: int = 5,
    has_emb_cand: bool = True,
    has_emb_hist: bool = True,
    with_target: bool = True,
    with_pos: bool = True,
) -> dict:
    """Генерирует синтетический сэмпл одной ленты с заданными параметрами."""
    rng = np.random.default_rng(feed_id)

    # Кандидаты
    cand_items = np.arange(1, m + 1, dtype=np.int64)
    cand_cats = rng.integers(1, 20, size=m, dtype=np.int64)
    if has_emb_cand:
        cand_embs = rng.normal(size=(m, EMB_DIM)).astype(np.float32)
        cand_has_emb = np.ones(m, dtype=bool)
    else:
        cand_embs = np.zeros((m, EMB_DIM), dtype=np.float32)
        cand_has_emb = np.zeros(m, dtype=bool)

    # Числовые фичи кандидатов f1..f5
    row_dummy = {f"f{i}": rng.normal(size=m).astype(np.float32) for i in range(1, 6)}

    # История
    if h > 0:
        h_items = rng.integers(1, 100, size=h, dtype=np.int64)
        h_cats = rng.integers(1, 20, size=h, dtype=np.int64)
        h_ev_types = rng.integers(1, 5, size=h, dtype=np.int64)
        h_ev_sources = rng.integers(1, 3, size=h, dtype=np.int64)
        if has_emb_hist:
            hist_embs = rng.normal(size=(h, EMB_DIM)).astype(np.float32)
            hist_has_emb = np.ones(h, dtype=bool)
        else:
            hist_embs = np.zeros((h, EMB_DIM), dtype=np.float32)
            hist_has_emb = np.zeros(h, dtype=bool)
    else:
        h_items = np.zeros(0, dtype=np.int64)
        h_cats = np.zeros(0, dtype=np.int64)
        h_ev_types = np.zeros(0, dtype=np.int64)
        h_ev_sources = np.zeros(0, dtype=np.int64)
        hist_embs = np.zeros((0, EMB_DIM), dtype=np.float32)
        hist_has_emb = np.zeros(0, dtype=bool)

    user_emb = rng.normal(size=EMB_DIM).astype(np.float32)
    cand_features = compute_candidate_features(
        row_dummy,
        cand_items,
        cand_cats,
        h_items,
        h_cats,
        user_emb=user_emb,
        cand_embs=cand_embs,
        cand_has_emb=cand_has_emb,
        hist_embs=hist_embs,
        hist_has_emb=hist_has_emb,
    )

    sample = {
        "feed_id": feed_id,
        "user_id": user_id,
        "user_emb": user_emb,
        "cand_len": m,
        "cand_item_id": cand_items,
        "cand_category_id": cand_cats,
        "cand_features": cand_features,
        "cand_emb": cand_embs,
        "cand_has_emb": cand_has_emb,
        "hist_len": h,
        "hist_item_id": h_items,
        "hist_category_id": h_cats,
        "hist_event_type": h_ev_types,
        "hist_event_source": h_ev_sources,
        "hist_emb": hist_embs,
        "hist_has_emb": hist_has_emb,
    }

    if with_target:
        sample["target"] = (rng.random(m) > 0.7).astype(np.float32)
    if with_pos:
        sample["pos"] = np.arange(m, dtype=np.int64)

    return sample


@pytest.fixture
def ranking_model() -> FeedRankingModel:
    model = FeedRankingModel(
        emb_dim=128,
        d_model=64,
        nhead=2,
        num_categories=100,
        num_cand_feats=NUM_CAND_FEATS,
        max_hist_len=256,
        dropout=0.1,
    )
    model.eval()
    return model


def test_empty_history(ranking_model):
    """1. Краевой случай: пустая история пользователя (H=0)."""
    s1 = make_dummy_sample(feed_id=1, m=10, h=0)
    s2 = make_dummy_sample(feed_id=2, m=12, h=0)

    batch = collate_ranking_batch([s1, s2])
    assert batch["history_mask"].sum() == 0, "Маска истории должна быть вся False при H=0"

    total_scores, main_scores = ranking_model(batch, use_pos_bias=True)

    # Проверка отсутствия NaN и Inf
    valid_mask = batch["candidate_mask"]
    assert not torch.isnan(main_scores[valid_mask]).any(), "main_scores содержит NaN при пустой истории!"
    assert not torch.isinf(main_scores[valid_mask]).any(), "main_scores содержит Inf при пустой истории!"
    assert not torch.isnan(total_scores[valid_mask]).any(), "total_scores содержит NaN при пустой истории!"

    # Проверка градиента (backward не должен выдавать NaN)
    ranking_model.train()
    total_scores, _ = ranking_model(batch, use_pos_bias=True)
    loss = total_scores[valid_mask].sum()
    loss.backward()
    for name, param in ranking_model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN в градиенте параметра {name} при H=0!"


def test_all_missing_embeddings(ranking_model):
    """2. Краевой случай: у всех кандидатов отсутствуют эмбеддинги (-1)."""
    s = make_dummy_sample(feed_id=1, m=15, h=10, has_emb_cand=False)
    batch = collate_ranking_batch([s])

    assert batch["cand_has_emb_mask"].sum() == 0, "cand_has_emb_mask должен быть весь False"
    assert torch.all(batch["cand_emb"] == 0.0), "cand_emb должен состоять из нулей"

    total_scores, main_scores = ranking_model(batch, use_pos_bias=False)
    valid_mask = batch["candidate_mask"]

    assert not torch.isnan(main_scores[valid_mask]).any(), "NaN при отсутствии эмбеддингов у кандидатов!"
    assert len(main_scores[valid_mask]) == 15


def test_single_candidate(ranking_model):
    """3. Краевой случай: лента ровно с одним кандидатом (M=1)."""
    s = make_dummy_sample(feed_id=1, m=1, h=5)
    batch = collate_ranking_batch([s])

    assert batch["candidate_mask"].shape == (1, 1)
    assert batch["candidate_mask"][0, 0] == True

    total_scores, main_scores = ranking_model(batch, use_pos_bias=True)
    assert main_scores.shape == (1, 1)
    assert not torch.isnan(main_scores).any()


def test_max_history_length(ranking_model):
    """4. Краевой случай: максимальная длина истории H=256."""
    s = make_dummy_sample(feed_id=1, m=20, h=256)
    batch = collate_ranking_batch([s])

    assert batch["history_mask"].shape[1] == 256
    assert batch["history_mask"].sum() == 256

    total_scores, main_scores = ranking_model(batch, use_pos_bias=False)
    valid_mask = batch["candidate_mask"]

    assert not torch.isnan(main_scores[valid_mask]).any(), "NaN при H=256!"
    assert len(main_scores[valid_mask]) == 20


def test_varying_candidate_sizes(ranking_model):
    """5. Краевой случай: ленты разного размера в одном батче (M=3, M=15, M=40)."""
    s1 = make_dummy_sample(feed_id=1, m=3, h=10)
    s2 = make_dummy_sample(feed_id=2, m=15, h=0)   # без истории
    s3 = make_dummy_sample(feed_id=3, m=40, h=50)  # длинная лента

    batch = collate_ranking_batch([s1, s2, s3])
    m_max = 40
    assert batch["candidate_mask"].shape == (3, m_max)

    # Проверка масок
    assert batch["candidate_mask"][0, :3].all() and not batch["candidate_mask"][0, 3:].any()
    assert batch["candidate_mask"][1, :15].all() and not batch["candidate_mask"][1, 15:].any()
    assert batch["candidate_mask"][2, :40].all()

    total_scores, main_scores = ranking_model(batch, use_pos_bias=True)

    # Паддинги должны быть строго замаскированы значением -1e9
    assert (main_scores[0, 3:] == -1e9).all()
    assert (main_scores[1, 15:] == -1e9).all()

    # Валидные кандидаты должны иметь конечные скоры без NaN
    valid_mask = batch["candidate_mask"]
    assert not torch.isnan(main_scores[valid_mask]).any()
    assert not torch.isinf(main_scores[valid_mask]).any()


def test_set_attention_padding_isolation_and_flag(ranking_model):
    """6. Проверка изоляции паддингов в Set-Attention и флага use_set_attention=False."""
    s_small = make_dummy_sample(feed_id=10, m=3, h=8)
    s_large = make_dummy_sample(feed_id=11, m=20, h=8)

    with torch.no_grad():
        # Прогон ленты на 3 элемента без паддинга
        b_solo = collate_ranking_batch([s_small])
        _, scores_solo = ranking_model(b_solo, use_pos_bias=False)

        # Прогон той же ленты с паддингом до 20 кандидатов
        b_padded = collate_ranking_batch([s_small, s_large])
        _, scores_padded = ranking_model(b_padded, use_pos_bias=False)

    assert torch.allclose(scores_solo[0], scores_padded[0, :3], atol=1e-5), (
        "Паддинги не должны влиять на предсказания Set-Attention!"
    )

    # Проверка модели без Set-Attention
    model_no_set = FeedRankingModel(
        emb_dim=128, d_model=64, nhead=2, num_categories=100,
        num_cand_feats=NUM_CAND_FEATS, use_set_attention=False,
    )
    model_no_set.eval()
    _, scores_no_set = model_no_set(b_padded, use_pos_bias=False)
    assert not torch.isnan(scores_no_set[b_padded["candidate_mask"]]).any()


def test_real_data_boundaries_and_clamp():
    """7. Проверка реальных границ из EDA (M=300, pos=299, event_source=30) и защиты clamp."""
    model = FeedRankingModel(
        emb_dim=128, d_model=64, nhead=2, num_categories=1000,
        num_cand_feats=NUM_CAND_FEATS, use_set_attention=True,
    )
    model.eval()

    s = make_dummy_sample(feed_id=99, m=300, h=50)
    # Эмулируем реальные границы и аномалии (выходящие за границы словаря)
    s["hist_event_source"][0] = 30
    s["hist_event_source"][1] = 999  # защита clamp
    s["hist_event_type"][0] = 10
    s["hist_event_type"][1] = 500    # защита clamp
    s["cand_category_id"][0] = 53
    s["cand_category_id"][1] = 5000  # защита clamp
    s["pos"] = np.arange(300, dtype=np.int64)  # 0..299
    s["pos"][-1] = 999  # защита clamp

    batch = collate_ranking_batch([s])
    with torch.no_grad():
        total_scores, main_scores = model(batch, use_pos_bias=True)

    assert not torch.isnan(total_scores).any()
    assert not torch.isnan(main_scores).any()
    assert total_scores.shape == (1, 300)
    # Проверяем, что pos=299 и pos=150 различаются (не схлопнуты на 100!)
    assert model.pos_bias_emb.num_embeddings == 512


def _write_mini_sharded_split(split_dir: Path, num_parts: int = 3, rows_per_part: int = 4) -> None:
    """Создает мини-сплит из нескольких parquet-партов и item_embeddings memmap."""
    imp_dir = split_dir / "impressions"
    emb_dir = split_dir / "item_embeddings"
    imp_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    num_embs = 64
    mm = np.memmap(emb_dir / "data.memmap", dtype="int8", mode="w+", shape=(num_embs, EMB_DIM))
    mm[:] = rng.integers(-64, 64, size=(num_embs, EMB_DIM), dtype=np.int8)
    mm.flush()
    del mm
    (emb_dir / "data.meta").write_text(
        json.dumps({"dtype": "int8", "shape": [num_embs, EMB_DIM], "order": "C"}),
        encoding="utf-8",
    )

    feed_id = 0
    for p_idx in range(num_parts):
        rows = []
        for r_idx in range(rows_per_part):
            m = 4 + (feed_id % 3)
            h = 3 if feed_id % 2 == 0 else 0
            # В одном из рядов pos=None для проверки mixed/null pos
            pos_val = None if (p_idx == 1 and r_idx == 0) else list(range(m))
            rows.append(
                {
                    "feed_id": feed_id,
                    "timestamp": 1000 + feed_id,
                    "user_id": 500 + feed_id,
                    "category_id": rng.integers(1, 20, size=m).tolist(),
                    "f1": rng.normal(size=m).tolist(),
                    "f2": rng.normal(size=m).tolist(),
                    "f3": rng.normal(size=m).tolist(),
                    "f4": rng.normal(size=m).tolist(),
                    "f5": rng.normal(size=m).tolist(),
                    "item_id__embedding_idx": [i if i % 3 != 0 else -1 for i in range(1, m + 1)],
                    "item_id": np.arange(100, 100 + m).tolist(),
                    "user_embedding": rng.integers(-30, 30, size=EMB_DIM).tolist(),
                    "history__item_id": rng.integers(1, 50, size=h).tolist(),
                    "history__category_id": rng.integers(1, 20, size=h).tolist(),
                    "history__event_type": rng.integers(1, 5, size=h).tolist(),
                    "history__event_source": rng.integers(0, 5, size=h).tolist(),
                    "history__item_id__embedding_idx": rng.integers(0, num_embs, size=h).tolist(),
                    "target": [1 if i == 0 else 0 for i in range(m)],
                    "pos": pos_val,
                }
            )
            feed_id += 1
        df = pl.DataFrame(rows)
        df.write_parquet(imp_dir / f"part-{p_idx:05d}.parquet")


def test_sharded_loader_parity_and_shuffle(tmp_path: Path):
    """Проверка эквивалентности ShardedFeedRankingDataset и Map-style FeedRankingDataset."""
    _write_mini_sharded_split(tmp_path / "train", num_parts=3, rows_per_part=4)

    map_ds = FeedRankingDataset(tmp_path, split="train")
    sharded_det = ShardedFeedRankingDataset(tmp_path, split="train", shuffle=False)

    assert len(sharded_det) == len(map_ds) == 12
    map_samples = [map_ds[i] for i in range(len(map_ds))]
    sharded_samples = list(sharded_det)
    assert [s["feed_id"] for s in sharded_samples] == [s["feed_id"] for s in map_samples]

    for ms, ss in zip(map_samples, sharded_samples):
        assert np.allclose(ms["cand_features"], ss["cand_features"])
        assert np.allclose(ms["cand_emb"], ss["cand_emb"])
        assert np.array_equal(ms["cand_has_emb"], ss["cand_has_emb"])
        assert np.array_equal(ms["target"], ss["target"])
        if "pos" in ms:
            assert np.array_equal(ms["pos"], ss["pos"])
        else:
            assert "pos" not in ss

    # Проверка shuffle и set_epoch
    sharded_rnd = ShardedFeedRankingDataset(tmp_path, split="train", shuffle=True, seed=42)
    sharded_rnd.set_epoch(1)
    ids_ep1 = [s["feed_id"] for s in sharded_rnd]
    sharded_rnd.set_epoch(2)
    ids_ep2 = [s["feed_id"] for s in sharded_rnd]

    assert sorted(ids_ep1) == list(range(12))
    assert sorted(ids_ep2) == list(range(12))
    assert ids_ep1 != ids_ep2

    # Проверка max_feeds
    limited = ShardedFeedRankingDataset(tmp_path, split="train", max_feeds=5, shuffle=True, seed=42)
    limited.set_epoch(1)
    assert len(limited) == 5
    assert len(list(limited)) == 5


def test_mixed_pos_debias_masking(ranking_model):
    """Проверка маскирования pos_bias при смешанном наличии pos (-1) в батче."""
    import copy

    s_with_pos = make_dummy_sample(feed_id=1, m=6, h=4, with_pos=True)
    s_no_pos = make_dummy_sample(feed_id=2, m=6, h=4, with_pos=False)

    batch = collate_ranking_batch([s_with_pos, s_no_pos])
    assert "pos" in batch
    assert (batch["pos"][0] >= 0).all()
    assert (batch["pos"][1] == -1).all()

    model = copy.deepcopy(ranking_model)
    model.eval()
    # Делаем большой pos_bias, чтобы сразу увидеть утечку при pos == -1
    model.pos_bias_emb.weight.data.fill_(100.0)

    with torch.no_grad():
        total_with_pos, main_with_pos = model(batch, use_pos_bias=True)
        total_no_bias, main_no_bias = model(batch, use_pos_bias=False)

    assert torch.isfinite(total_with_pos).all()
    assert torch.allclose(main_with_pos, main_no_bias, atol=1e-6)
    # На сэмпле 0 pos >= 0 -> pos_bias ненулевой (одинаковый ненулевой сдвиг, т.к. weights=100)
    bias_0 = total_with_pos[0] - main_with_pos[0]
    assert torch.all(bias_0.abs() > 0.5)
    # На сэмпле 1 pos == -1 -> pos_bias должен быть строго занулен!
    assert torch.allclose(total_with_pos[1], total_no_bias[1], atol=1e-6)


def test_cosine_features_math_and_recent_window():
    """Проверка математики 9 cosine-фичей Димы, окна recent=10 и размерностей MLP."""
    assert NUM_CAND_FEATS == 24
    expected_9 = [
        "cos_user",
        "cos_hist_mean",
        "cos_hist_max",
        "cos_hist_recent_max",
        "cos_hist_recent_mean",
        "item_norm",
        "cat_in_recent",
        "item_hist_first_pos",
        "has_hist_emb_mask_rate",
    ]
    assert CAND_FEAT_NAMES[-9:] == expected_9

    # Проверка размерностей MLP в FeedRankingModel по умолчанию
    model = FeedRankingModel()
    assert model.cand_encoder.mlp[0].in_features == 128 + 32 + 24
    assert model.main_head[0].in_features == 128 * 3 + 3 + 1 + 24

    # 3 кандидата: [0] вдоль оси 0 (norm=2.0), [1] вдоль оси 1 (norm=3.0), [2] без эмбеддинга (-1)
    user_emb = np.zeros(EMB_DIM, dtype=np.float32)
    user_emb[0] = 5.0

    cand_embs = np.zeros((3, EMB_DIM), dtype=np.float32)
    cand_embs[0, 0] = 2.0
    cand_embs[1, 1] = 3.0
    cand_has_emb = np.array([True, True, False], dtype=bool)
    cand_items = np.array([101, 102, 103], dtype=np.int64)
    cand_cats = np.array([7, 8, 9], dtype=np.int64)

    # История длиной 11: первые 10 событий без эмбеддинга, 11-е (индекс 10) = [-1, 0, ...]
    h = 11
    h_items = np.arange(200, 211, dtype=np.int64)
    h_items[4] = 102  # кандидат 102 впервые на позиции 4
    h_cats = np.full(h, 99, dtype=np.int64)
    h_cats[0] = 7
    h_cats[1] = 7  # 2 из 10 recent категорий == 7

    hist_embs = np.zeros((h, EMB_DIM), dtype=np.float32)
    hist_embs[10, 0] = -4.0
    hist_has_emb = np.zeros(h, dtype=bool)
    hist_has_emb[10] = True

    row_dummy = {f"f{i}": np.zeros(3, dtype=np.float32) for i in range(1, 6)}
    feats = compute_candidate_features(
        row_dummy,
        cand_items,
        cand_cats,
        h_items,
        h_cats,
        user_emb=user_emb,
        cand_embs=cand_embs,
        cand_has_emb=cand_has_emb,
        hist_embs=hist_embs,
        hist_has_emb=hist_has_emb,
    )
    assert feats.shape == (3, 24)
    assert np.isfinite(feats).all()

    idx = {name: i for i, name in enumerate(CAND_FEAT_NAMES)}
    # cos_user: [1.0, 0.0, 0.0]
    assert np.allclose(feats[:, idx["cos_user"]], [1.0, 0.0, 0.0], atol=1e-5)
    # item_norm: [2.0, 3.0, 0.0]
    assert np.allclose(feats[:, idx["item_norm"]], [2.0, 3.0, 0.0], atol=1e-5)
    # Первые 10 событий без эмб -> recent_max и recent_mean строго 0.0 (не подменяются 11-м событием!)
    assert np.allclose(feats[:, idx["cos_hist_recent_max"]], 0.0, atol=1e-6)
    assert np.allclose(feats[:, idx["cos_hist_recent_mean"]], 0.0, atol=1e-6)
    # Вся история видит 11-е событие [-1, 0, ...]
    assert np.allclose(feats[:, idx["cos_hist_max"]], [-1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feats[:, idx["cos_hist_mean"]], [-1.0, 0.0, 0.0], atol=1e-5)
    # has_hist_emb_mask_rate: 1 / 11
    assert np.allclose(feats[:, idx["has_hist_emb_mask_rate"]], 1.0 / 11.0, atol=1e-5)
    # cat_in_recent: [2/10, 0, 0]
    assert np.allclose(feats[:, idx["cat_in_recent"]], [0.2, 0.0, 0.0], atol=1e-5)
    # item_hist_first_pos: [-1.0, 4/256.0, -1.0]
    assert np.allclose(feats[:, idx["item_hist_first_pos"]], [-1.0, 4.0 / 256.0, -1.0], atol=1e-6)

    # Добавляем валидный эмбеддинг [1, 0, ...] в окно recent (индекс 0)
    hist_embs[0, 0] = 1.0
    hist_has_emb[0] = True
    feats2 = compute_candidate_features(
        row_dummy,
        cand_items,
        cand_cats,
        h_items,
        h_cats,
        user_emb=user_emb,
        cand_embs=cand_embs,
        cand_has_emb=cand_has_emb,
        hist_embs=hist_embs,
        hist_has_emb=hist_has_emb,
    )
    assert np.allclose(feats2[:, idx["cos_hist_recent_max"]], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feats2[:, idx["cos_hist_recent_mean"]], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feats2[:, idx["cos_hist_max"]], [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feats2[:, idx["cos_hist_mean"]], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(feats2[:, idx["has_hist_emb_mask_rate"]], 2.0 / 11.0, atol=1e-5)
