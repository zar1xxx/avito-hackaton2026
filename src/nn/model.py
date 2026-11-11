"""Продвинутая нейросетевая модель ранжирования рекомендаций Avito.

Архитектура:
1. History Transformer Encoder:
   - Товарные эмбеддинги (memmap) + category_id + event_type + event_source + positional_embedding.
   - Multi-Head Self-Attention слой с маскированием паддингов (history_mask).
2. Candidate Encoder:
   - Товарные эмбеддинги + category_id + числовые признаки (f1..f5) -> MLP в d_model.
3. Candidate-to-History Cross-Attention:
   - Query: кандидаты [B, M, d_model]
   - Key / Value: история [B, H, d_model]
   - Находит релевантные события истории для каждого отдельного кандидата.
4. User Interaction & Ranking Head:
   - Объединяет представление пользователя, кандидата, контекста истории и их dot-product взаимодействия.
5. Positional Debias Head (Two-Tower Debias):
   - Отдельная голова для моделирования position bias по фиче `pos` во время train.
   - На валидации/тесте голова позиции отключается, гарантируя несмещенный релевантный скор.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_hist_len: int = 256,
        num_categories: int = 1000,
        num_event_types: int = 16,
        num_event_sources: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Эмбеддинги признаков истории
        self.cat_emb = nn.Embedding(num_categories, 32, padding_idx=0)
        self.ev_type_emb = nn.Embedding(num_event_types, 16, padding_idx=0)
        self.ev_source_emb = nn.Embedding(num_event_sources, 16, padding_idx=0)
        self.pos_emb = nn.Embedding(max_hist_len + 1, 32)

        in_dim = 128 + 32 + 16 + 16 + 32
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        hist_emb: torch.Tensor,
        hist_cat: torch.Tensor,
        hist_ev_type: torch.Tensor,
        hist_ev_source: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        hist_emb: [B, H, 128]
        hist_cat, hist_ev_type, hist_ev_source: [B, H]
        history_mask: [B, H] (bool, True = валидное событие)
        
        Returns:
            encoded_history: [B, H, d_model]
        """
        b, h, _ = hist_emb.shape

        # Позиции в истории: 0 = самое свежее, 1, 2...
        pos_idx = torch.arange(h, device=hist_emb.device).unsqueeze(0).expand(b, h)
        pos_idx = torch.clamp(pos_idx, 0, self.pos_emb.num_embeddings - 1)

        c_emb = self.cat_emb(torch.clamp(hist_cat, min=0))
        et_emb = self.ev_type_emb(torch.clamp(hist_ev_type, min=0))
        es_emb = self.ev_source_emb(torch.clamp(hist_ev_source, min=0))
        p_emb = self.pos_emb(pos_idx)

        x = torch.cat([hist_emb, c_emb, et_emb, es_emb, p_emb], dim=-1)
        x = self.input_proj(x)  # [B, H, d_model]

        # Защита от NaN: если у пользователя вообще нет истории (вся строка mask = False),
        # временно открываем первый токен, а затем зануляем выход
        has_any = history_mask.any(dim=1, keepdim=True)  # [B, 1]
        safe_mask = history_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | (~has_any.squeeze(1))

        # В PyTorch src_key_padding_mask: True = игнорировать позицию
        padding_mask = ~safe_mask

        encoded = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, H, d_model]

        # Зануляем паддинги
        encoded = encoded * history_mask.unsqueeze(-1).float()
        return encoded


class CandidateEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_categories: int = 1000,
        num_cand_feats: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories, 32, padding_idx=0)
        in_dim = 128 + 32 + num_cand_feats

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        cand_emb: torch.Tensor,
        cand_cat: torch.Tensor,
        cand_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        cand_emb: [B, M, 128]
        cand_cat: [B, M]
        cand_feats: [B, M, num_cand_feats]
        
        Returns:
            [B, M, d_model]
        """
        cat_e = self.cat_emb(torch.clamp(cand_cat, min=0))
        x = torch.cat([cand_emb, cat_e, cand_feats], dim=-1)
        return self.mlp(x)


class FeedRankingModel(nn.Module):
    """Полнофункциональная модель ранжирования ленты с Cross-Attention и дебиасингом."""

    def __init__(
        self,
        emb_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_categories: int = 1000,
        num_cand_feats: int = 15,
        max_hist_len: int = 256,
        max_position: int = 100,
        dropout: float = 0.15,
        use_set_attention: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_set_attention = use_set_attention

        # 1. Энкодер истории
        self.hist_encoder = HistoryTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=1,
            dropout=dropout,
            max_hist_len=max_hist_len,
            num_categories=num_categories,
        )

        # 2. Энкодер кандидатов
        self.cand_encoder = CandidateEncoder(
            d_model=d_model,
            num_categories=num_categories,
            num_cand_feats=num_cand_feats,
            dropout=dropout,
        )

        # 3. Проекция пользователя
        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # 4. Cross-Attention: Query = Кандидаты, Key/Value = История
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # 4.5. Listwise Set-Attention: Self-Attention между кандидатами внутри ленты
        if self.use_set_attention:
            self.set_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.set_norm = nn.LayerNorm(d_model)
            self.set_dropout = nn.Dropout(dropout)

        # 5. Основная голова скоринга (Контентная релевантность)
        head_in_dim = d_model * 3 + 3 + 1 + num_cand_feats
        self.main_head = nn.Sequential(
            nn.Linear(head_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 6. Голова дебиасинга позиции (Position Bias Tower)
        # Обучается только на train по фиче pos
        self.pos_bias_emb = nn.Embedding(max_position, 32)
        self.pos_bias_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        use_pos_bias: bool = False,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            batch: словарь тензоров от collate_ranking_batch.
            use_pos_bias: если True и в batch есть 'pos', к скору добавляется position bias.
                          На val/test должно быть False!
            return_features: если True, дополнительно возвращает словарь косинусных сходств.
                          
        Returns:
            scores: [B, M] — итоговые скоры для обучения (с позицией, если use_pos_bias=True).
            main_scores: [B, M] — чистые контентные скоры без позиции (для валидации и сабмита).
        """
        user_emb = batch["user_emb"]            # [B, 128]
        cand_emb = batch["cand_emb"]            # [B, M, 128]
        cand_cat = batch["cand_category_id"]    # [B, M]
        cand_feats = batch["cand_features"]     # [B, M, num_feats]
        cand_mask = batch["candidate_mask"]     # [B, M]
        cand_has_emb = batch["cand_has_emb_mask"] # [B, M]

        b, m, _ = cand_emb.shape

        # 1. Энкодинг пользователя
        user_repr = self.user_proj(user_emb)    # [B, d_model]

        # 2. Энкодинг кандидатов
        cand_repr = self.cand_encoder(cand_emb, cand_cat, cand_feats) # [B, M, d_model]

        # 3. Энкодинг истории
        history_mask = batch["history_mask"]    # [B, H]
        hist_encoded = self.hist_encoder(
            hist_emb=batch["hist_emb"],
            hist_cat=batch["hist_category_id"],
            hist_ev_type=batch["hist_event_type"],
            hist_ev_source=batch["hist_event_source"],
            history_mask=history_mask,
        ) # [B, H, d_model]

        # 4. Cross-Attention (Query = cand_repr, Key/Value = hist_encoded)
        has_any_hist = history_mask.any(dim=1, keepdim=True) # [B, 1]
        safe_hist_mask = history_mask.clone()
        safe_hist_mask[:, 0] = safe_hist_mask[:, 0] | (~has_any_hist.squeeze(1))
        padding_mask = ~safe_hist_mask

        attn_out, _ = self.cross_attn(
            query=cand_repr,
            key=hist_encoded,
            value=hist_encoded,
            key_padding_mask=padding_mask,
        ) # [B, M, d_model]

        attn_out = attn_out * has_any_hist.unsqueeze(-1).float()
        cand_hist_ctx = self.cross_norm(cand_repr + attn_out) # Residual

        # 4.5. Listwise Set-Attention между кандидатами внутри ленты
        if self.use_set_attention:
            has_any_cand = cand_mask.any(dim=1, keepdim=True) # [B, 1]
            safe_cand_mask = cand_mask.clone()
            safe_cand_mask[:, 0] = safe_cand_mask[:, 0] | (~has_any_cand.squeeze(1))

            set_out, _ = self.set_attn(
                query=cand_hist_ctx,
                key=cand_hist_ctx,
                value=cand_hist_ctx,
                key_padding_mask=~safe_cand_mask,
            ) # [B, M, d_model]
            set_out = set_out * cand_mask.unsqueeze(-1).float()
            cand_hist_ctx = self.set_norm(cand_hist_ctx + self.set_dropout(set_out))
            cand_hist_ctx = cand_hist_ctx * cand_mask.unsqueeze(-1).float()

        # 5. Скалярные произведения и сходства
        u_norm = F.normalize(user_repr, p=2, dim=-1).unsqueeze(1) # [B, 1, d_model]
        c_norm = F.normalize(cand_repr, p=2, dim=-1)             # [B, M, d_model]
        h_norm = F.normalize(cand_hist_ctx, p=2, dim=-1)         # [B, M, d_model]

        dot_u = (c_norm * u_norm).sum(dim=-1, keepdim=True)      # [B, M, 1]
        dot_h = (c_norm * h_norm).sum(dim=-1, keepdim=True)      # [B, M, 1]
        dot_uh = (u_norm * h_norm).sum(dim=-1, keepdim=True)     # [B, M, 1]
        has_emb_f = cand_has_emb.unsqueeze(-1).float()           # [B, M, 1]

        # 6. Основной контентный скор (Main Head)
        user_rep_exp = user_repr.unsqueeze(1).expand(-1, m, -1)
        head_input = torch.cat([
            cand_hist_ctx,
            user_rep_exp,
            cand_repr,
            dot_u,
            dot_h,
            dot_uh,
            has_emb_f,
            cand_feats,
        ], dim=-1)

        main_scores = self.main_head(head_input).squeeze(-1) # [B, M]

        # 7. Positional Debias Tower
        if use_pos_bias and "pos" in batch and batch["pos"] is not None:
            pos_indices = torch.clamp(batch["pos"], 0, self.pos_bias_emb.num_embeddings - 1)
            pos_emb = self.pos_bias_emb(pos_indices) # [B, M, 32]
            pos_bias = self.pos_bias_head(pos_emb).squeeze(-1) # [B, M]
            total_scores = main_scores + pos_bias
        else:
            total_scores = main_scores

        # Маскируем паддинги большим отрицательным значением
        total_scores = total_scores.masked_fill(~cand_mask, -1e9)
        main_scores = main_scores.masked_fill(~cand_mask, -1e9)

        if return_features:
            feats_dict = {
                "dot_user": dot_u.squeeze(-1).masked_fill(~cand_mask, 0.0),
                "dot_hist": dot_h.squeeze(-1).masked_fill(~cand_mask, 0.0),
                "dot_uh": dot_uh.squeeze(-1).masked_fill(~cand_mask, 0.0),
            }
            return total_scores, main_scores, feats_dict

        return total_scores, main_scores
