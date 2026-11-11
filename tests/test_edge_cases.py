"""Unit-тесты для проверки устойчивости модели и датасета к краевым случаям (edge cases).

Проверяет:
1. Пользователь без истории (H=0). Отсутствие NaN / Inf в скорах и градиентах.
2. Все кандидаты в ленте без эмбеддингов (item_id__embedding_idx = -1).
3. Лента с ровно одним кандидатом (M=1).
4. Максимальная длина истории (H=256).
5. Разные размеры лент в одном батче (динамический паддинг и маски).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from src.nn.dataset import (
    EMB_DIM,
    NUM_CAND_FEATS,
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

    cand_features = compute_candidate_features(row_dummy, cand_items, cand_cats, h_items, h_cats)

    sample = {
        "feed_id": feed_id,
        "user_id": user_id,
        "user_emb": rng.normal(size=EMB_DIM).astype(np.float32),
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
