"""
aim_bridge/stage1_diagnosis.py
==============================
Stage 1: 感知缺口診斷與符號穩定性驗證

核心目標：
  在完全不修改 V-JEPA 2 權重的前提下，
  驗證其潛空間是否具備足夠的線性可分性來支持 AIM 符號化。

執行流程：
  1. 載入凍結的 V-JEPA 2 encoder
  2. 訓練輕量 AIM Quantizer（Stage A，encoder 完全凍結）
  3. 量化器收斂後切換 eval 模式
  4. H1 符號穩定性測試
  5. H2 干預實驗（三個物理變數）+ 隨機基線對照
  6. 輸出：通過標準報告 + AIM 字典初版 + 診斷圖表

通過標準（進入 Stage 2 的門檻）：
  H1 符號穩定性    > 95%
  H2 chi² p 值    < 0.01
  H2 MI 強度      實驗組 > 隨機基線 × 5
  Codebook 利用率  > 30%

使用方式：
  python stage1_diagnosis.py \\
      --encoder_checkpoint /path/to/vjepa2_vitg.pt \\
      --video_root /path/to/intervention_videos \\
      --codebook_size 64 \\
      --output_dir ./stage1_results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon

# 對接已有模組（所有 .py 在同一目錄下）
sys.path.append(str(Path(__file__).parent))

from aim_dictionary_json import AIMDictionary
from aim_intervention_builder import (
    InterventionExperiment,
    MutualInfoAnalyzer,
    AIMDictionaryBuilder,
    CrossValidator,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 專用 AIM Quantizer
# 輕量設計：只訓練這一個元件，encoder 完全凍結
# ═══════════════════════════════════════════════════════════════════════

class Stage1AIMQuantizer(nn.Module):
    """
    Stage 1 專用的輕量 AIM 量化器。

    設計決策：
    - 單層 VQ（不用殘差量化），降低 Stage 1 的複雜度
    - 1408 → projection_dim 的投影層，降低高維空間方差影響
    - EMA codebook 更新，比 gradient update 更穩定
    - L2 normalization 在 temporal pooling 之後、投影之前執行

    Stage 1 不需要多層 VQ，目標只是驗證「符號能否對應物理語義」，
    單層已足夠提供統計檢驗所需的信號。
    """

    def __init__(
        self,
        input_dim: int = 1408,
        projection_dim: int = 256,
        codebook_size: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # 投影層：降維並壓縮高方差空間
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),  # 穩定投影後的數值範圍
        )

        # Codebook（EMA 更新，不參與 gradient）
        self.register_buffer(
            "codebook", F.normalize(torch.randn(codebook_size, projection_dim), dim=-1)
        )
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.codebook.clone())
        self.register_buffer("usage_counter", torch.zeros(codebook_size))

    def encode(self, z_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_frame: [B, D]  ← temporal pooling 之後的 frame-level 向量
        Returns: (z_q, indices, loss)
        """
        B, D = z_frame.shape

        # L2 Normalization（在投影前做，穩定高維輸入）
        # 注意：在 temporal pooling 之後才做，保留幀間相對尺度
        z_norm = F.normalize(z_frame, dim=-1)

        # 投影到低維空間
        z_proj = self.projection(z_norm)           # [B, projection_dim]
        z_proj_norm = F.normalize(z_proj, dim=-1)  # 投影後再 normalize

        # 計算與 codebook 的距離
        dist = (
            z_proj_norm.pow(2).sum(1, keepdim=True)
            - 2 * z_proj_norm @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )  # [B, K]

        indices = dist.argmin(1)   # [B]
        z_q = self.codebook[indices]  # [B, projection_dim]

        # EMA 更新（只在 training 模式）
        if self.training:
            self._ema_update(z_proj_norm, indices)

        # Commitment loss（讓 encoder 輸出靠近 codebook）
        loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_proj_norm)

        # Straight-through estimator
        z_q_st = z_proj_norm + (z_q - z_proj_norm).detach()

        # 更新使用計數
        self.usage_counter.scatter_add_(
            0, indices, torch.ones(B, device=indices.device)
        )

        return z_q_st, indices, loss

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.codebook_size).float()
        cluster_size = one_hot.sum(0)
        embed_sum = one_hot.t() @ z_flat

        self.ema_cluster_size = (
            self.ema_decay * self.ema_cluster_size
            + (1 - self.ema_decay) * cluster_size
        )
        self.ema_embed_sum = (
            self.ema_decay * self.ema_embed_sum
            + (1 - self.ema_decay) * embed_sum
        )
        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.codebook_size * 1e-5) * n
        )
        updated = self.ema_embed_sum / smoothed.unsqueeze(1)
        self.codebook = F.normalize(updated, dim=-1)

    def compute_perplexity(self) -> float:
        total = self.usage_counter.sum()
        if total == 0:
            return 0.0
        probs = self.usage_counter / total
        return float(torch.exp(-torch.sum(probs * torch.log(probs + 1e-10))).item())

    def active_code_ratio(self) -> float:
        return float((self.usage_counter > 0).float().mean().item())

    def reset_dead_codes(self, z_batch: torch.Tensor, threshold_ratio: float = 0.01):
        """重置使用率過低的死碼"""
        usage = self.usage_counter / (self.usage_counter.sum() + 1e-10)
        dead = usage < (threshold_ratio / self.codebook_size)
        n_dead = dead.sum().item()
        if n_dead > 0 and len(z_batch) > 0:
            z_norm = F.normalize(z_batch, dim=-1)
            z_proj = self.projection(z_norm)
            z_proj_norm = F.normalize(z_proj, dim=-1)
            variances = torch.var(z_proj_norm, dim=1)
            
            n_replace = min(int(n_dead), len(variances))
            if n_replace < n_dead:
                dead_indices = torch.where(dead)[0][:n_replace]
                dead = torch.zeros_like(dead, dtype=torch.bool)
                dead[dead_indices] = True
                n_dead = n_replace
                
            top_idx = variances.topk(n_dead).indices
            new_codes = z_proj_norm[top_idx].detach()
            
            self.codebook[dead] = new_codes
            self.usage_counter[dead] = 0
            self.ema_cluster_size[dead] = 0.0
            self.ema_embed_sum[dead] = new_codes.clone()
        return int(n_dead)


# ═══════════════════════════════════════════════════════════════════════
# Temporal Pooling 工具函數
# ═══════════════════════════════════════════════════════════════════════

def temporal_pool(z: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    V-JEPA 2 encoder 輸出：z ∈ [B, N_tokens, D]
    其中 N_tokens = num_frames × N_spatial

    正確做法：先把空間維度 pool 掉，只保留時間維度
    返回：[B, num_frames, D]

    注意：必須在 L2 normalization 之前執行，
    保留各幀之間的相對尺度信息。
    """
    B, N_tokens, D = z.shape
    assert N_tokens % num_frames == 0, (
        f"N_tokens ({N_tokens}) must be divisible by num_frames ({num_frames})"
    )
    N_spatial = N_tokens // num_frames
    z_reshaped = z.reshape(B, num_frames, N_spatial, D)
    z_frame = z_reshaped.mean(dim=2)   # [B, num_frames, D]
    return z_frame


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 核心診斷器
# ═══════════════════════════════════════════════════════════════════════

class Stage1Diagnostician:
    """
    Stage 1 全流程執行器。

    三個主要任務：
    A. 量化器訓練（Stage A warm-up）
    B. 符號穩定性測試（H1）
    C. 干預實驗 + 隨機基線（H2）
    """

    # 通過標準
    PASS_CRITERIA = {
        "h1_consistency":    0.95,   # 符號穩定性 > 95%
        "h2_chi2_p":         0.01,   # p < 0.01
        "h2_mi_ratio":       5.0,    # 實驗組 MI > 基線 × 5
        "codebook_active":   0.30,   # 活躍符號 > 30%
    }

    def __init__(
        self,
        encoder,                        # 凍結的 V-JEPA 2 encoder
        codebook_size: int = 64,
        projection_dim: int = 256,
        num_frames: int = 16,
        device: str = "cuda",
        output_dir: str = "./stage1_results",
        encoder_embed_dim: int = 1408,  # ViT-g
    ):
        self.encoder = encoder.eval()
        self.num_frames = num_frames
        self.device = device
        self.output_dir = Path(output_dir)
        self.codebook_size = codebook_size

        # 凍結 encoder（所有參數不參與梯度）
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        log.info("Encoder frozen. No gradients will flow through encoder.")

        # 初始化量化器
        self.quantizer = Stage1AIMQuantizer(
            input_dim=encoder_embed_dim,
            projection_dim=projection_dim,
            codebook_size=codebook_size,
        ).to(device)

        self.aim_dict = AIMDictionary(
            str(self.output_dir / "aim_dictionary_stage1.json")
        )
        self.analyzer = MutualInfoAnalyzer(codebook_size)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        # 診斷報告
        self.report: Dict = {
            "stage": "Stage1",
            "timestamp": datetime.now().isoformat(),
            "criteria": self.PASS_CRITERIA,
            "results": {},
            "passed": False,
        }

    # ──────────────────────────────────────────────────────────────────
    # Encoder 輔助函數：統一處理維度順序和呼叫方式
    # ──────────────────────────────────────────────────────────────────



    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        統一的 encoder 呼叫介面，處理所有可能的輸入維度格式。

        接受的輸入格式：
          [B, C, T, H, W]  ← DataLoader 正確輸出，直接用
          [B, T, C, H, W]  ← 需要 permute
          [C, T, H, W]     ← 單樣本無 batch 維度，加上 unsqueeze(0)
          [1, C, T, H, W]  ← 單樣本帶 batch 維度，直接用

        Returns: last_hidden_state [B, N_tokens, D]
        """
        # Step 1：統一成 5 維
        if video.dim() == 4:
            video = video.unsqueeze(0)   # [C,T,H,W] → [1,C,T,H,W]
        elif video.dim() == 6:
            video = video.squeeze(1)     # [B,1,C,T,H,W] → [B,C,T,H,W]

        # Step 2：準備兩種格式 (HF 需要 [B, T, C, H, W], Local 需要 [B, C, T, H, W])
        if video.shape[1] != 3 and video.shape[2] == 3:
            # 目前是 [B, T, C, H, W]
            video_hf = video
            video_local = video.permute(0, 2, 1, 3, 4).contiguous()
        else:
            # 目前是 [B, C, T, H, W]
            video_hf = video.permute(0, 2, 1, 3, 4).contiguous()
            video_local = video

        # Step 3：呼叫 encoder
        try:
            out = self.encoder(pixel_values_videos=video_hf)
            return out.last_hidden_state   # [B, N_tokens, D]
        except TypeError:
            out = self.encoder(video_local)
            if hasattr(out, 'last_hidden_state'):
                return out.last_hidden_state
            return out

    # ──────────────────────────────────────────────────────────────────
    # Task A：量化器訓練（Stage A warm-up）
    # ──────────────────────────────────────────────────────────────────

    def train_quantizer(
        self,
        video_loader,                   # DataLoader，輸出 (video_tensor, label)
        warmup_iterations: int = 3000,
        lr: float = 1e-3,
        perplexity_target_ratio: float = 0.4,  # 健康閾值：log(K) 的 40%
        dead_code_reset_interval: int = 200,
    ):
        """
        Stage A：只訓練量化器，encoder 完全凍結。

        收斂標準：
        - Perplexity / log(K) > 0.4（健康碼本）
        - 連續 200 iteration loss 不再下降
        """
        log.info(f"=== Stage A: Quantizer Warm-up ===")
        log.info(f"  Target: {warmup_iterations} iterations")
        log.info(f"  Perplexity health threshold: {perplexity_target_ratio:.0%} of log({self.codebook_size})")

        optimizer = optim.Adam(self.quantizer.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=warmup_iterations, eta_min=lr * 0.1
        )

        max_perplexity = np.log(self.codebook_size)
        health_threshold = max_perplexity * perplexity_target_ratio

        loss_history = []
        perplexity_history = []
        collected_z_for_reset = []

        self.quantizer.train()
        step = 0
        converged = False

        while step < warmup_iterations and not converged:
            for video_batch, _ in video_loader:
                if step >= warmup_iterations:
                    break

                video_batch = video_batch.to(self.device)

                # Encoder forward（no_grad）
                with torch.no_grad():
                    z = self._encode(video_batch)                     # [B, N_tokens, D]
                    z_frames = temporal_pool(z, self.num_frames)      # [B, T, D]

                # 對每一幀量化
                total_loss = torch.tensor(0.0, device=self.device)
                B, T, D = z_frames.shape
                batch_z_list = []

                for t in range(T):
                    z_t = z_frames[:, t, :]   # [B, D]
                    batch_z_list.append(z_t)
                    _, _, loss_t = self.quantizer.encode(z_t)
                    total_loss += loss_t

                total_loss /= T
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                # 收集用於死碼重置的向量
                collected_z_for_reset.append(
                    torch.cat(batch_z_list, dim=0).detach()
                )
                if len(collected_z_for_reset) > 10:
                    collected_z_for_reset.pop(0)

                # 監控
                perp = self.quantizer.compute_perplexity()
                active = self.quantizer.active_code_ratio()
                loss_history.append(total_loss.item())
                perplexity_history.append(perp)

                if step % 100 == 0:
                    log.info(
                        f"  Step {step:4d} | Loss={total_loss.item():.4f} | "
                        f"Perplexity={perp:.2f}/{max_perplexity:.2f} "
                        f"({perp/max_perplexity:.0%}) | "
                        f"Active={active:.0%}"
                    )

                # 死碼重置
                if step % dead_code_reset_interval == 0 and step > 0:
                    z_pool = torch.cat(collected_z_for_reset, dim=0)
                    n_reset = self.quantizer.reset_dead_codes(z_pool)
                    if n_reset > 0:
                        log.info(f"  ↻ Reset {n_reset} dead codes at step {step}")

                # 收斂判斷
                if step >= 500 and perp >= health_threshold:
                    recent_loss = np.mean(loss_history[-200:])
                    older_loss = np.mean(loss_history[-400:-200])
                    if abs(recent_loss - older_loss) / (older_loss + 1e-8) < 0.01:
                        log.info(
                            f"  ✅ Converged at step {step} "
                            f"(perplexity={perp:.2f}, health={perp/max_perplexity:.0%})"
                        )
                        converged = True
                        break

                step += 1

        # 切換到 eval 模式——H1 測試必須在 eval 模式下進行
        self.quantizer.eval()
        final_perp = self.quantizer.compute_perplexity()
        final_active = self.quantizer.active_code_ratio()

        log.info(f"\n  Final perplexity: {final_perp:.2f} / {max_perplexity:.2f} "
                 f"({final_perp/max_perplexity:.0%})")
        log.info(f"  Final active codes: {final_active:.0%}")

        self._plot_training_curve(loss_history, perplexity_history, max_perplexity)

        self.report["results"]["quantizer_training"] = {
            "final_perplexity": round(final_perp, 3),
            "final_perplexity_ratio": round(final_perp / max_perplexity, 3),
            "final_active_ratio": round(final_active, 3),
            "total_steps": step,
            "converged": converged,
            "codebook_healthy": final_perp / max_perplexity >= perplexity_target_ratio,
        }

        return converged

    # ──────────────────────────────────────────────────────────────────
    # Task B：H1 符號穩定性測試
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test_h1_stability(
        self,
        stability_videos: List[torch.Tensor],  # 5-10 個影片樣本
        n_repeats: int = 20,
    ) -> float:
        """
        H1 測試：同一影片重複輸入 n_repeats 次，計算符號一致性率。

        必須在量化器 eval 模式下執行（確保 codebook 不再更新）。
        理論上 VQ 是確定性的，eval 模式下應接近 100%。
        若低於 95%，說明預處理（如 random augmentation）引入了隨機性。
        """
        assert not self.quantizer.training, \
            "Must be in eval mode for stability test!"

        log.info(f"\n=== H1: Symbol Stability Test ({n_repeats} repeats) ===")

        all_consistency = []

        for vid_idx, video in enumerate(stability_videos):
            if video.dim() == 4:
                video = video.unsqueeze(0)
            video = video.to(self.device)
            self.encoder.eval()  # 確保每次測試時 encoder 都在 eval 模式
            # 重複量化 n_repeats 次
            all_symbols = []
            for _ in range(n_repeats):
                z = self._encode(video)
                z_frames = temporal_pool(z, self.num_frames)
                frame_symbols = []
                for t in range(self.num_frames):
                    _, idx, _ = self.quantizer.encode(z_frames[:, t, :])
                    frame_symbols.append(idx[0].item())
                all_symbols.append(frame_symbols)

            # 計算每個時間步的一致性率
            consistency_per_frame = []
            for t in range(self.num_frames):
                symbols_at_t = [run[t] for run in all_symbols]
                most_common = max(set(symbols_at_t), key=symbols_at_t.count)
                rate = symbols_at_t.count(most_common) / n_repeats
                consistency_per_frame.append(rate)

            video_consistency = np.mean(consistency_per_frame)
            all_consistency.append(video_consistency)
            log.info(f"  Video {vid_idx}: consistency={video_consistency:.3f}")

        mean_consistency = np.mean(all_consistency)
        passed = mean_consistency >= self.PASS_CRITERIA["h1_consistency"]

        log.info(f"\n  H1 Result: {mean_consistency:.3f} "
                 f"(threshold={self.PASS_CRITERIA['h1_consistency']}) "
                 f"{'✅ PASS' if passed else '❌ FAIL'}")

        self.report["results"]["h1_stability"] = {
            "mean_consistency": round(mean_consistency, 4),
            "per_video": [round(c, 4) for c in all_consistency],
            "threshold": self.PASS_CRITERIA["h1_consistency"],
            "passed": passed,
        }

        return mean_consistency

    # ──────────────────────────────────────────────────────────────────
    # Task C：H2 干預實驗 + 隨機基線
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_symbols_for_condition(
        self,
        videos: List[torch.Tensor],
    ) -> List[List[int]]:
        """收集一批影片的符號序列（eval 模式，no_grad）"""
        all_sequences = []
        for video in videos:
            if video.dim() == 4:
                video = video.unsqueeze(0)
            video = video.to(self.device)
            z = self._encode(video)
            z_frames = temporal_pool(z, self.num_frames)
            symbols = []
            for t in range(self.num_frames):
                _, idx, _ = self.quantizer.encode(z_frames[:, t, :])
                symbols.append(idx[0].item())
            all_sequences.append(symbols)
        return all_sequences

    @torch.no_grad()
    def generate_random_baseline(
        self,
        n_samples: int,
    ) -> InterventionExperiment:
        """
        生成隨機基線：用高斯雜訊輸入，模擬 MI ≈ 0 的情況。
        這是最重要的對照組——如果你的實驗組 MI 不顯著高於這個，
        說明符號只是反映了 codebook 的隨機結構，不是物理語義。
        """
        log.info(f"\n  Generating random baseline ({n_samples} samples × 2 conditions)...")

        # 用兩組不同種子的噪聲作為「偽條件」
        baseline_exp = InterventionExperiment(
            variable_name="RANDOM_BASELINE",
            conditions=["noise_seed_A", "noise_seed_B"],
            condition_labels=["noise_A", "noise_B"],
            num_frames=self.num_frames,
        )

        for cond_idx, seed in enumerate([42, 1337]):
            torch.manual_seed(seed)
            for _ in range(n_samples):
                # noise 使用 [B, C, T, H, W] 格式，與真實影片一致
                noise = torch.randn(1, 3, self.num_frames, 224, 224).to(self.device)
                z = self._encode(noise)
                z_frames = temporal_pool(z, self.num_frames)
                symbols = []
                for t in range(self.num_frames):
                    _, idx, _ = self.quantizer.encode(z_frames[:, t, :])
                    symbols.append(idx[0].item())
                baseline_exp.record(cond_idx, symbols)

        return baseline_exp

    def run_intervention_experiment(
        self,
        variable_name: str,
        condition_videos: Dict[str, List[torch.Tensor]],
        condition_labels: Optional[List[str]] = None,
    ) -> Dict:
        """
        針對單一物理變數執行完整干預實驗。

        condition_videos: {"0deg": [video1, video2, ...], "30deg": [...], ...}
        """
        if condition_labels is None:
            condition_labels = list(condition_videos.keys())

        conditions = list(condition_videos.keys())
        log.info(f"\n=== Intervention: {variable_name} ({len(conditions)} conditions) ===")

        exp = InterventionExperiment(
            variable_name=variable_name,
            conditions=conditions,
            condition_labels=condition_labels,
            num_frames=self.num_frames,
        )

        # 收集各條件的符號
        for cond_idx, (cond_key, videos) in enumerate(condition_videos.items()):
            sequences = self.collect_symbols_for_condition(videos)
            for seq in sequences:
                exp.record(cond_idx, seq)
            log.info(f"  Condition '{cond_key}': {len(sequences)} samples collected")

        # 計算統計指標
        mi_score = self.analyzer.compute_mi(exp)
        chi2, p_val, is_sig = self.analyzer.chi2_significance_test(exp)
        jsd_matrix = self.analyzer.compute_pairwise_jsd(exp)
        sensitivity = self.analyzer.compute_per_symbol_sensitivity(exp)
        mapping = self.analyzer.find_condition_symbol_mapping(exp)

        mean_jsd = float(np.mean(jsd_matrix[jsd_matrix > 0])) \
            if jsd_matrix.any() else 0.0

        log.info(f"  MI={mi_score:.4f} | chi² p={p_val:.4f} "
                 f"{'✅' if is_sig else '❌'} | mean JSD={mean_jsd:.3f}")
        log.info(f"  Condition→Symbol: {mapping}")

        # 寫入字典
        if is_sig and mi_score > 0:
            for cond_idx, label in enumerate(condition_labels):
                dominant = mapping.get(label, -1)
                if dominant < 0:
                    continue
                evidence = {
                    "experiment": f"stage1_intervention_{variable_name}",
                    "MI_score": round(mi_score, 4),
                    "chi2_p_value": round(p_val, 4),
                    "is_significant": is_sig,
                    "mean_JSD": round(mean_jsd, 4),
                    "condition_value": conditions[cond_idx],
                    "symbol_mapping": mapping,
                    "stage": "Stage1_Frozen_Encoder",
                    "timestamp": datetime.now().isoformat(),
                }
                self.aim_dict.add_entry(
                    aim_id_sequence_str=str([dominant]),
                    human_label=f"{variable_name}={label}",
                    context=json.dumps(evidence, sort_keys=True),
                )

        # 生成視覺化
        self._plot_intervention(
            exp, jsd_matrix, sensitivity, variable_name, mapping
        )

        result = {
            "variable": variable_name,
            "mi_score": round(mi_score, 4),
            "chi2_p_value": round(p_val, 6),
            "is_significant": is_sig,
            "mean_jsd": round(mean_jsd, 4),
            "condition_symbol_mapping": mapping,
            "n_samples_per_condition": {
                condition_labels[i]: len(exp.symbol_records[i])
                for i in range(len(conditions))
            },
        }
        return result

    def run_full_diagnosis(
        self,
        # 干預實驗資料
        variable_A_videos: Dict[str, List[torch.Tensor]],  # 抓握角度
        variable_B_videos: Dict[str, List[torch.Tensor]],  # 物體幾何
        variable_C_videos: Dict[str, List[torch.Tensor]],  # 運動速度
        # H1 測試資料
        stability_videos: List[torch.Tensor],
        # 量化器訓練資料
        train_loader,
        # 設定
        warmup_iterations: int = 3000,
        n_baseline_samples: int = 30,
    ) -> Dict:
        """
        Stage 1 完整流程：A → H1 → H2 → 報告
        """
        log.info("\n" + "="*60)
        log.info("STAGE 1: Perception Gap Diagnosis & Symbol Stability")
        log.info("="*60)

        # ── Task A：量化器訓練 ──────────────────────────────────────
        converged = self.train_quantizer(train_loader, warmup_iterations)
        if not converged:
            log.warning("Quantizer did not converge! Consider increasing warmup_iterations.")

        # ── Task B：H1 穩定性測試 ────────────────────────────────────
        h1_score = self.test_h1_stability(stability_videos)

        # ── Task C：H2 干預實驗 ──────────────────────────────────────
        results_A = self.run_intervention_experiment("grasp_angle",    variable_A_videos)
        results_B = self.run_intervention_experiment("object_geometry", variable_B_videos)
        results_C = self.run_intervention_experiment("motion_speed",   variable_C_videos)

        # ── 隨機基線 ─────────────────────────────────────────────────
        baseline_exp = self.generate_random_baseline(n_baseline_samples)
        baseline_mi = self.analyzer.compute_mi(baseline_exp)
        log.info(f"\n  Random baseline MI: {baseline_mi:.4f}")

        # ── MI ratio 計算 ────────────────────────────────────────────
        experiment_mis = [results_A["mi_score"],
                         results_B["mi_score"],
                         results_C["mi_score"]]
        mi_ratios = [
            mi / (baseline_mi + 1e-8)
            for mi in experiment_mis
        ]

        # ── Codebook 健康狀態 ─────────────────────────────────────────
        final_active = self.quantizer.active_code_ratio()
        final_perp = self.quantizer.compute_perplexity()

        # ── 通過標準評估 ──────────────────────────────────────────────
        criteria_results = {
            "h1_consistency": {
                "value": round(h1_score, 4),
                "threshold": self.PASS_CRITERIA["h1_consistency"],
                "passed": h1_score >= self.PASS_CRITERIA["h1_consistency"],
            },
            "h2_chi2_p": {
                "values": {
                    "grasp_angle":    results_A["chi2_p_value"],
                    "object_geometry": results_B["chi2_p_value"],
                    "motion_speed":   results_C["chi2_p_value"],
                },
                "threshold": self.PASS_CRITERIA["h2_chi2_p"],
                "passed": all(r["chi2_p_value"] < self.PASS_CRITERIA["h2_chi2_p"]
                              for r in [results_A, results_B, results_C]),
            },
            "h2_mi_ratio": {
                "values": {
                    "grasp_angle":    round(mi_ratios[0], 2),
                    "object_geometry": round(mi_ratios[1], 2),
                    "motion_speed":   round(mi_ratios[2], 2),
                },
                "baseline_mi": round(baseline_mi, 4),
                "threshold": self.PASS_CRITERIA["h2_mi_ratio"],
                "passed": all(r >= self.PASS_CRITERIA["h2_mi_ratio"]
                              for r in mi_ratios),
            },
            "codebook_active": {
                "value": round(final_active, 4),
                "perplexity": round(final_perp, 3),
                "threshold": self.PASS_CRITERIA["codebook_active"],
                "passed": final_active >= self.PASS_CRITERIA["codebook_active"],
            },
        }

        all_passed = all(v["passed"] for v in criteria_results.values())

        # ── 儲存 ──────────────────────────────────────────────────────
        self.aim_dict.save()
        self.report["results"].update({
            "intervention_experiments": {
                "grasp_angle":    results_A,
                "object_geometry": results_B,
                "motion_speed":   results_C,
            },
            "random_baseline_mi": round(baseline_mi, 4),
            "criteria_results": criteria_results,
        })
        self.report["passed"] = all_passed

        # 儲存報告
        report_path = self.output_dir / "stage1_report.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        # ── 列印摘要 ──────────────────────────────────────────────────
        self._print_summary(criteria_results, all_passed)

        # 生成彙總圖
        self._plot_summary(criteria_results, mi_ratios,
                           [results_A, results_B, results_C])

        return self.report

    # ──────────────────────────────────────────────────────────────────
    # 視覺化
    # ──────────────────────────────────────────────────────────────────

    def _plot_training_curve(
        self, loss_history, perplexity_history, max_perplexity
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Stage A: Quantizer Training", fontweight="bold")

        axes[0].plot(loss_history, color="#3498db", lw=1.2)
        axes[0].set_title("Commitment Loss")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")

        axes[1].plot(perplexity_history, color="#2ecc71", lw=1.2)
        axes[1].axhline(max_perplexity * 0.4, ls="--", color="#e74c3c",
                       lw=1.0, label=f"Health threshold (40% of log({self.codebook_size}))")
        axes[1].axhline(max_perplexity, ls=":", color="gray",
                       lw=0.8, label=f"Max perplexity (log({self.codebook_size})={max_perplexity:.2f})")
        axes[1].set_title("Codebook Perplexity")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Perplexity")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "stage_a_training.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_intervention(
        self, exp, jsd_matrix, sensitivity, var_name, mapping
    ):
        n_conds = len(exp.conditions)
        labels = exp.condition_labels

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Intervention: {var_name}", fontsize=13, fontweight="bold")

        # 圖1：符號分佈
        top_syms = sorted(sensitivity, key=sensitivity.get, reverse=True)[:20]
        x = np.arange(len(top_syms))
        width = 0.8 / n_conds
        colors = plt.cm.Set2(np.linspace(0, 1, n_conds))
        for ci, (label, color) in enumerate(zip(labels, colors)):
            dist = exp.get_symbol_distribution(ci, self.codebook_size)
            axes[0].bar(x + ci * width, [dist[s] for s in top_syms],
                       width, label=label, color=color, alpha=0.8)
        axes[0].set_xticks(x + width * (n_conds-1)/2)
        axes[0].set_xticklabels([str(s) for s in top_syms], rotation=45, fontsize=7)
        axes[0].set_title("Symbol Distribution (top-20)")
        axes[0].legend(fontsize=8)

        # 圖2：JSD 熱力圖
        im = axes[1].imshow(jsd_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        axes[1].set_xticks(range(n_conds))
        axes[1].set_yticks(range(n_conds))
        axes[1].set_xticklabels(labels, rotation=45, fontsize=9)
        axes[1].set_yticklabels(labels, fontsize=9)
        plt.colorbar(im, ax=axes[1])
        for i in range(n_conds):
            for j in range(n_conds):
                axes[1].text(j, i, f"{jsd_matrix[i,j]:.2f}",
                           ha="center", va="center", fontsize=8,
                           color="white" if jsd_matrix[i,j] > 0.5 else "black")
        axes[1].set_title("Pairwise JSD")

        # 圖3：敏感度排名
        top30 = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:30]
        if top30:
            syms, scores = zip(*top30)
            mean_score = np.mean(scores)
            bar_colors = ["#e74c3c" if sc > mean_score * 2 else "#3498db"
                         for sc in scores]
            axes[2].bar(range(len(syms)), scores, color=bar_colors)
            axes[2].set_xticks(range(len(syms)))
            axes[2].set_xticklabels([str(s) for s in syms], rotation=45, fontsize=7)

            # 標注 mapping 中的符號
            mapped_syms = set(mapping.values())
            for i, s in enumerate(syms):
                if s in mapped_syms:
                    axes[2].get_xticklabels()[i].set_color("#e74c3c")
                    axes[2].get_xticklabels()[i].set_fontweight("bold")
        axes[2].set_title("Symbol Sensitivity (red=in mapping)")

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / f"intervention_{var_name}.png",
                   dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_summary(self, criteria_results, mi_ratios, experiment_results):
        """彙總圖：論文 Figure 2 的基礎"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Stage 1 Diagnosis Summary", fontsize=13, fontweight="bold")

        # 圖1：MI ratio（實驗組 vs 基線）
        var_names = ["grasp_angle", "object_geometry", "motion_speed"]
        colors = ["#2ecc71" if r >= self.PASS_CRITERIA["h2_mi_ratio"]
                 else "#e74c3c" for r in mi_ratios]
        bars = axes[0].bar(var_names, mi_ratios, color=colors, alpha=0.8)
        axes[0].axhline(self.PASS_CRITERIA["h2_mi_ratio"], ls="--",
                       color="#e74c3c", lw=1.5, label=f"Threshold (×{self.PASS_CRITERIA['h2_mi_ratio']})")
        axes[0].set_title("MI Ratio (Experiment / Baseline)")
        axes[0].set_ylabel("MI Ratio")
        axes[0].legend()
        for bar, ratio in zip(bars, mi_ratios):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f"×{ratio:.1f}", ha="center", va="bottom", fontsize=9)

        # 圖2：chi² p 值（對數尺度）
        p_values = [r["chi2_p_value"] for r in experiment_results]
        p_colors = ["#2ecc71" if p < self.PASS_CRITERIA["h2_chi2_p"]
                   else "#e74c3c" for p in p_values]
        axes[1].bar(var_names, [-np.log10(p + 1e-10) for p in p_values],
                   color=p_colors, alpha=0.8)
        axes[1].axhline(-np.log10(self.PASS_CRITERIA["h2_chi2_p"]),
                       ls="--", color="#e74c3c", lw=1.5,
                       label=f"p={self.PASS_CRITERIA['h2_chi2_p']} threshold")
        axes[1].set_title("Chi-square Significance (-log₁₀ p)")
        axes[1].set_ylabel("-log₁₀(p-value)")
        axes[1].legend()

        # 圖3：通過標準雷達圖（用長條圖近似）
        criteria_names = list(criteria_results.keys())
        pass_status = [1 if criteria_results[k]["passed"] else 0
                      for k in criteria_names]
        bar_colors = ["#2ecc71" if p else "#e74c3c" for p in pass_status]
        axes[2].bar(criteria_names, pass_status, color=bar_colors, alpha=0.8)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["FAIL", "PASS"])
        axes[2].set_title("Criteria Pass/Fail Summary")
        axes[2].tick_params(axis='x', rotation=20)

        overall = "✅ READY FOR STAGE 2" if all(pass_status) else "❌ NEEDS REVIEW"
        fig.text(0.5, 0.01, overall, ha="center", fontsize=12,
                fontweight="bold",
                color="#2ecc71" if all(pass_status) else "#e74c3c")

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(self.output_dir / "figures" / "stage1_summary.png",
                   dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"\n  📊 Summary figure saved.")

    def _print_summary(self, criteria_results, all_passed):
        log.info("\n" + "="*60)
        log.info("STAGE 1 DIAGNOSIS RESULTS")
        log.info("="*60)
        for key, result in criteria_results.items():
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            if "value" in result:
                log.info(f"  {key:25s}: {result['value']:.4f} "
                        f"(threshold={result['threshold']}) {status}")
            else:
                log.info(f"  {key:25s}: {status}")
                for subkey, val in result.get("values", {}).items():
                    log.info(f"    {subkey}: {val}")
        log.info("-"*60)
        if all_passed:
            log.info("  🚀 ALL CRITERIA PASSED → Ready to proceed to Stage 2")
        else:
            log.info("  ⚠️  Some criteria failed → Review before Stage 2")
            log.info("  Suggestions:")
            if not criteria_results["codebook_active"]["passed"]:
                log.info("    - Increase warmup_iterations or lower commitment_cost")
            if not criteria_results["h1_consistency"]["passed"]:
                log.info("    - Check if video preprocessing has random augmentation")
            if not criteria_results["h2_chi2_p"]["passed"]:
                log.info("    - Increase samples per condition (target: 50+)")
            if not criteria_results["h2_mi_ratio"]["passed"]:
                log.info("    - Verify physical conditions are sufficiently distinct")
        log.info("="*60)


# ═══════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: AIM-JEPA Diagnosis")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                       help="Path to V-JEPA 2 pretrained checkpoint (.pt file)")
    parser.add_argument("--video_root", type=str,
                       default="./data/kinetics_mini/val",
                       help="Root directory of intervention videos")
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--warmup_iterations", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="./stage1_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder_embed_dim", type=int, default=1024,
                       help="1024 for ViT-L (default), 1408 for ViT-g")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_per_class", type=int, default=50,
                       help="Max videos per class to load")
    parser.add_argument("--run_mock", action="store_true",
                       help="Run with mock encoder for testing (no checkpoint needed)")
    parser.add_argument("--run_with_kinetics", action="store_true",
                       help="Run with real Kinetics-mini videos + real encoder")
    return parser.parse_args()


def run_mock_test(args):
    """
    Mock 測試：不需要真實 V-JEPA 2 checkpoint。
    用來驗證整個流程可以跑通。
    設計了有差異的 mock encoder，
    讓不同條件的潛向量具有統計上可區分的特性。
    """
    log.info("Running MOCK test (no real checkpoint needed)")

    class MockEncoder(nn.Module):
        """Mock encoder：不同條件的影片輸入會產生不同偏移的輸出"""
        def __init__(self):
            super().__init__()

        def forward(self, x):
            B = x.shape[0]
            pixel_mean = x.mean(dim=(1, 2, 3, 4)).view(B, 1).to(x.device)
            
            freqs = torch.arange(args.encoder_embed_dim, device=x.device).float()
            offset = torch.sin(pixel_mean * freqs * 10.0).view(B, 1, -1) * 3.0
            
            z = torch.randn(B, args.num_frames * 14 * 14, args.encoder_embed_dim, device=x.device) * 0.5
            return z + offset

    encoder = MockEncoder()

    # 準備 mock 影片資料
    def make_videos(n, brightness_offset=0.0):
        return [
            torch.randn(1, 3, args.num_frames, 64, 64) + brightness_offset
            for _ in range(n)
        ]

    variable_A = {
        "0deg":  make_videos(30, 0.0),
        "30deg": make_videos(30, 0.5),
        "60deg": make_videos(30, 1.0),
        "90deg": make_videos(30, 1.5),
    }
    variable_B = {
        "sphere":   make_videos(30, 0.2),
        "cube":     make_videos(30, 0.7),
        "cylinder": make_videos(30, 1.2),
    }
    variable_C = {
        "slow":   make_videos(25, 0.1),
        "normal": make_videos(25, 0.6),
        "fast":   make_videos(25, 1.1),
    }
    stability_videos = make_videos(5, 0.3)
    all_videos = (
        [v for vlist in variable_A.values() for v in vlist] +
        [v for vlist in variable_B.values() for v in vlist]
    )

    from torch.utils.data import DataLoader, TensorDataset
    all_tensors = torch.cat([v for v in all_videos], dim=0)
    dummy_labels = torch.zeros(len(all_tensors), dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(all_tensors, dummy_labels),
        batch_size=8, shuffle=True
    )

    diagnostician = Stage1Diagnostician(
        encoder=encoder,
        codebook_size=args.codebook_size,
        projection_dim=args.projection_dim,
        num_frames=args.num_frames,
        device="cpu",
        output_dir=args.output_dir,
        encoder_embed_dim=args.encoder_embed_dim,
    )

    report = diagnostician.run_full_diagnosis(
        variable_A_videos=variable_A,
        variable_B_videos=variable_B,
        variable_C_videos=variable_C,
        stability_videos=stability_videos,
        train_loader=train_loader,
        warmup_iterations=min(args.warmup_iterations, 500),
    )

    return report


def run_kinetics_test(args):
    """
    真實執行路徑：V-JEPA 2 encoder + Kinetics-mini 影片。

    執行前確認：
    1. checkpoints/vjepa2_vitg.pt 已下載
       wget https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -O checkpoints/vjepa2_vitg.pt
    2. data/kinetics_mini/val/ 已下載
       python -c "
       from huggingface_hub import snapshot_download
       snapshot_download(repo_id='nateraw/kinetics-mini', repo_type='dataset',
           local_dir='./data/kinetics_mini', allow_patterns='val/*/*.mp4')
       "
    """
    from video_dataset import (
        build_condition_dict,
        build_train_loader,
        check_environment,
    )

    # 環境檢查
    if not check_environment():
        log.error("Environment check failed. Please install missing packages.")
        return None

    # 載入 V-JEPA 2 encoder
    log.info(f"Loading V-JEPA 2 encoder...")
    try:
        from transformers import AutoModel, AutoVideoProcessor
        hf_repo = "facebook/vjepa2-vitl-fpc64-256"   # ViT-L（較小，適合測試）
        if args.encoder_embed_dim == 1408:
            hf_repo = "facebook/vjepa2-vitg-fpc64-256"  # ViT-g

        processor = AutoVideoProcessor.from_pretrained(
            hf_repo,
            do_normalize=True,       # 保留正規化（encoder 需要）
            do_resize=True,          # 保留 resize（encoder 需要）
            do_center_crop=True,     # 用中心裁切取代隨機裁切
            do_random_crop=False,    # ← 關掉隨機裁切
            do_flip=False,           # ← 關掉隨機翻轉
            do_color_jitter=False,   # ← 關掉顏色抖動
        )
        encoder   = AutoModel.from_pretrained(hf_repo).to(args.device)
        encoder.eval()

        # 確認所有 dropout 和 batchnorm 也進入推理模式
        for module in encoder.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm1d,
                                   nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
        log.info("  Encoder fully in eval mode (dropout/BN disabled)")

        log.info(f"  Loaded from HuggingFace: {hf_repo}")



    except Exception as e:
        import traceback
        log.warning(f"HuggingFace load failed: {e}")
        log.warning(traceback.format_exc())
        log.warning("Trying local checkpoint...")

        if args.encoder_checkpoint and Path(args.encoder_checkpoint).exists():
            encoder   = torch.load(args.encoder_checkpoint, map_location=args.device)
            processor = None
            encoder.eval()
            for module in encoder.modules():
                if isinstance(module, (nn.Dropout, nn.BatchNorm1d,
                                       nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
            log.info("  Encoder fully in eval mode (dropout/BN disabled)")

        else:
            log.error("No encoder available. Download with:")
            log.error("  wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt "
                     "-O checkpoints/vjepa2_vitl.pt")
            return None

    # 載入 Kinetics-mini 影片
    log.info(f"\nLoading Kinetics-mini from: {args.video_root}")
    data = build_condition_dict(
        video_root=args.video_root,
        processor=processor,
        device=args.device,
        num_frames=args.num_frames,
        max_per_class=args.max_per_class,
    )

    train_loader = build_train_loader(
        video_root=args.video_root,
        processor=processor,
        device=args.device,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
    )

    # 初始化診斷器
    diagnostician = Stage1Diagnostician(
        encoder=encoder,
        codebook_size=args.codebook_size,
        projection_dim=args.projection_dim,
        num_frames=args.num_frames,
        device=args.device,
        output_dir=args.output_dir,
        encoder_embed_dim=args.encoder_embed_dim,
    )

    # 執行完整診斷
    report = diagnostician.run_full_diagnosis(
        variable_A_videos=data["variable_A"],
        variable_B_videos=data["variable_B"],
        variable_C_videos=data["variable_C"],
        stability_videos=data["stability"],
        train_loader=train_loader,
        warmup_iterations=args.warmup_iterations,
    )

    return report


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_mock:
        # Mock 測試：不需要任何 checkpoint 或影片
        # 用途：確認程式碼流程正確
        log.info("="*50)
        log.info("Running MOCK test (no checkpoint/video needed)")
        log.info("="*50)
        report = run_mock_test(args)

    elif args.run_with_kinetics:
        # 真實執行：V-JEPA 2 + Kinetics-mini
        log.info("="*50)
        log.info("Running with real encoder + Kinetics-mini videos")
        log.info("="*50)
        report = run_kinetics_test(args)

    else:
        log.info("Usage:")
        log.info("")
        log.info("  # Step 1：先用 mock 確認流程（不需要任何資料）")
        log.info("  python stage1_diagnosis.py --run_mock")
        log.info("")
        log.info("  # Step 2：下載 Kinetics-mini 影片")
        log.info("  python -c \"")
        log.info("  from huggingface_hub import snapshot_download")
        log.info("  snapshot_download(repo_id='nateraw/kinetics-mini',")
        log.info("      repo_type='dataset', local_dir='./data/kinetics_mini',")
        log.info("      allow_patterns='val/*/*.mp4')\"")
        log.info("")
        log.info("  # Step 3：真實執行")
        log.info("  python stage1_diagnosis.py --run_with_kinetics \\")
        log.info("      --video_root ./data/kinetics_mini/val \\")
        log.info("      --output_dir ./stage1_results")
        log.info("")
        log.info("  # 完整參數說明：")
        log.info("  python stage1_diagnosis.py --help")
