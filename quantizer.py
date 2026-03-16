"""
AIM × V-JEPA 2 核心量化器
整合 hqvae_components.py 的層次 VQ 機制
適配 V-JEPA 2 的 1408 維 ViT-g 潛空間
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import sys
sys.path.append('../aim')
from hqvae_components import HierarchicalVQVAE  # 直接使用你的實作


class AIMQuantizerForVJEPA(nn.Module):
    """
    AIM 量化器，專為 V-JEPA 2 的潛空間設計。

    核心設計原則：
    1. 殘差量化（Residual VQ）：三層逐級壓縮，保留多尺度語義
    2. EMA codebook 更新：比 gradient update 更穩定
    3. Perplexity 監控：實時偵測 codebook collapse
    4. 動態死碼重置：防止符號退化
    """

    def __init__(
        self,
        input_dim: int = 1408,       # ViT-g embed_dim
        num_levels: int = 3,          # 殘差量化層數
        codebook_sizes: List[int] = [64, 128, 256],  # 每層詞彙量
        commitment_cost: float = 0.25,
        decay: float = 0.99,          # EMA 衰減率
        dead_code_threshold: float = 1.0,  # perplexity 低於此值時重置
        projection_dim: int = 256,    # 投影到較低維度再量化
    ):
        super().__init__()
        assert len(codebook_sizes) == num_levels
        self.num_levels = num_levels
        self.codebook_sizes = codebook_sizes
        self.commitment_cost = commitment_cost
        self.dead_code_threshold = dead_code_threshold

        # 投影層：1408 → 256，降維後再量化（減少計算量）
        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.output_projection = nn.Linear(projection_dim, input_dim)

        # 每一層的 VQ codebook（EMA 更新）
        self.codebooks = nn.ModuleList([
            VQCodebook(
                num_embeddings=k,
                embedding_dim=projection_dim,
                decay=decay,
                commitment_cost=commitment_cost
            )
            for k in codebook_sizes
        ])

        # 監控用的使用率統計
        self.register_buffer('usage_counts',
            torch.zeros(sum(codebook_sizes)))

    def forward(
        self,
        z: torch.Tensor,              # [B, N_tokens, 1408]
        return_indices: bool = True,
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            z_q:      量化後向量 [B, N_tokens, 1408]，可直接送入 predictor
            indices:  各層符號索引 list of [B, N_tokens]
            vq_loss:  量化訓練損失
        """
        B, N, D = z.shape

        # 投影到低維空間
        z_proj = self.input_projection(z)  # [B, N, 256]
        z_proj_flat = z_proj.reshape(B * N, -1)  # [B*N, 256]

        # 殘差量化：每層量化「剩餘殘差」
        residual = z_proj_flat.clone()
        quantized_levels = []
        indices_levels = []
        total_vq_loss = 0.0

        for level, codebook in enumerate(self.codebooks):
            z_q_level, idx_level, loss_level = codebook(residual)
            quantized_levels.append(z_q_level)
            indices_levels.append(idx_level.reshape(B, N))
            total_vq_loss += loss_level

            # 殘差：下一層只學沒被當前層捕捉到的資訊
            residual = residual - z_q_level.detach()

        # 重建：所有層的量化向量加總
        z_q_combined = sum(quantized_levels)  # [B*N, 256]
        z_q_combined = z_q_combined.reshape(B, N, -1)

        # 投影回原始維度
        z_q_out = self.output_projection(z_q_combined)  # [B, N, 1408]

        # Straight-through estimator：反向傳播用原始梯度
        z_q_out = z + (z_q_out - z).detach()

        return z_q_out, indices_levels, total_vq_loss / self.num_levels

    def compute_perplexity(self, indices: torch.Tensor, codebook_size: int) -> float:
        """計算 codebook 使用分佈的 perplexity，越高代表越均勻（越健康）"""
        one_hot = F.one_hot(indices.flatten(), codebook_size).float()
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        ).item()
        return perplexity

    def reset_dead_codes(self, z_batch: torch.Tensor, level: int):
        """
        重置使用率過低的死碼
        用當前 batch 的高方差特徵替換
        """
        codebook = self.codebooks[level]
        usage = codebook.get_usage()
        K = self.codebook_sizes[level]
        dead_mask = usage < (self.dead_code_threshold / K)

        if dead_mask.any():
            n_dead = dead_mask.sum().item()
            z_flat = self.input_projection(z_batch).reshape(-1, z_batch.shape[-1])
            # 選方差最大的特徵向量作為新碼
            variances = torch.var(z_flat, dim=1)
            _, top_idx = variances.topk(min(n_dead, len(variances)))
            new_codes = z_flat[top_idx[:n_dead]].detach()
            codebook.reset_codes(dead_mask, new_codes)
            return n_dead
        return 0


class VQCodebook(nn.Module):
    """單層 VQ Codebook，使用 EMA 更新（比 gradient 更穩定）"""

    def __init__(self, num_embeddings, embedding_dim, decay=0.99,
                 commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook 向量
        self.register_buffer('embeddings',
            torch.randn(num_embeddings, embedding_dim))
        self.embeddings = F.normalize(self.embeddings, dim=-1)

        # EMA 統計量
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_sum',
            self.embeddings.clone())
        self.register_buffer('usage_counter', torch.zeros(num_embeddings))

    def forward(self, z_flat):
        # z_flat: [B*N, D]
        # 計算與所有碼向量的距離
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embeddings.t()
            + self.embeddings.pow(2).sum(1)
        )
        indices = dist.argmin(1)  # [B*N]

        # 查表得到量化向量
        z_q = self.embeddings[indices]  # [B*N, D]

        # EMA 更新（只在 training 時更新）
        if self.training:
            self._ema_update(z_flat, indices)

        # Commitment loss
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_flat)

        # Straight-through
        z_q_st = z_flat + (z_q - z_flat).detach()

        # 更新使用計數
        self.usage_counter.scatter_add_(
            0, indices, torch.ones_like(indices, dtype=torch.float))

        return z_q_st, indices, loss

    def _ema_update(self, z_flat, indices):
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        cluster_size = one_hot.sum(0)
        embed_sum = one_hot.t() @ z_flat

        self.ema_cluster_size = (
            self.decay * self.ema_cluster_size
            + (1 - self.decay) * cluster_size
        )
        self.ema_embed_sum = (
            self.decay * self.ema_embed_sum
            + (1 - self.decay) * embed_sum
        )

        # 正規化更新
        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.num_embeddings * 1e-5) * n
        )
        self.embeddings = self.ema_embed_sum / smoothed.unsqueeze(1)

    def get_usage(self):
        total = self.usage_counter.sum()
        if total == 0:
            return torch.zeros_like(self.usage_counter)
        return self.usage_counter / total

    def reset_codes(self, dead_mask, new_codes):
        self.embeddings[dead_mask] = new_codes
        self.usage_counter[dead_mask] = 0