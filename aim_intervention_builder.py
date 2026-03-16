"""
aim_intervention_builder.py
============================
AIM Dictionary Builder via Controlled Intervention Experiments

功能：
  1. InterventionExperiment   - 針對單一物理變數的受控干預實驗
  2. MutualInfoAnalyzer       - 計算符號與變數的互資訊，量化相關性
  3. AIMDictionaryBuilder     - 整合上述兩者，自動生成有因果證據的字典條目
  4. CrossValidator           - 用新場景驗證字典的泛化能力

對接關係：
  - 輸入：JEPA encoder 輸出的連續向量 z（任何維度）
  - 量化：AIMQuantizerForVJEPA（或任何 VQ 量化器）
  - 輸出：直接寫入 AIMDictionary（aim_dictionary_json.py）

使用方式：
    builder = AIMDictionaryBuilder(encoder, quantizer, aim_dict)
    builder.run_experiment("grasp_angle", conditions, videos_per_condition)
    builder.finalize_and_save()
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable, Any
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

# 對接 AIMDictionary

from aim_dictionary_json import AIMDictionary


# ═══════════════════════════════════════════════════════════════════════
# Step 1: 受控干預實驗
# ═══════════════════════════════════════════════════════════════════════

class InterventionExperiment:
    """
    針對單一物理變數進行受控干預實驗。

    設計原則：
    - 只改變目標變數（intervention variable）
    - 固定所有其他條件（control variables）
    - 收集每個條件下的符號分佈

    這對應 Pearl 的 do-calculus：do(X = x_i) 並觀察符號 S 的分佈變化。
    """

    def __init__(
        self,
        variable_name: str,
        conditions: List[Any],           # 各條件的標籤，如 [0, 30, 60, 90]（角度）
        condition_labels: List[str],     # 人類可讀標籤，如 ["0deg","30deg","60deg","90deg"]
        num_frames: int = 16,            # V-JEPA 2 預設幀數
    ):
        self.variable_name = variable_name
        self.conditions = conditions
        self.condition_labels = condition_labels
        self.num_frames = num_frames

        # 收集結果：{condition_idx: [symbol_sequence, ...]}
        self.symbol_records: Dict[int, List[List[int]]] = defaultdict(list)
        # 原始向量記錄（用於後續分析）
        self.latent_records: Dict[int, List[np.ndarray]] = defaultdict(list)

    def record(
        self,
        condition_idx: int,
        symbols: List[int],         # 該樣本的符號序列（已量化）
        z_vector: Optional[np.ndarray] = None,  # 對應的潛空間向量（可選）
    ):
        """記錄一個樣本的符號輸出"""
        self.symbol_records[condition_idx].append(symbols)
        if z_vector is not None:
            self.latent_records[condition_idx].append(z_vector)

    def get_symbol_distribution(
        self, condition_idx: int, codebook_size: int
    ) -> np.ndarray:
        """取得某個條件下的符號頻率分佈（K 維向量）"""
        all_symbols = []
        for seq in self.symbol_records[condition_idx]:
            all_symbols.extend(seq)
        if not all_symbols:
            return np.ones(codebook_size) / codebook_size  # 無資料時返回均勻分佈
        counts = np.bincount(all_symbols, minlength=codebook_size).astype(float)
        counts += 1e-8  # Laplace smoothing
        return counts / counts.sum()

    def get_dominant_symbol(self, condition_idx: int) -> int:
        """取得某個條件下最高頻的符號"""
        all_symbols = []
        for seq in self.symbol_records[condition_idx]:
            all_symbols.extend(seq)
        if not all_symbols:
            return -1
        return Counter(all_symbols).most_common(1)[0][0]

    def summary(self) -> Dict:
        """實驗結果摘要"""
        return {
            "variable": self.variable_name,
            "n_conditions": len(self.conditions),
            "samples_per_condition": {
                self.condition_labels[i]: len(self.symbol_records[i])
                for i in range(len(self.conditions))
            },
            "dominant_symbols": {
                self.condition_labels[i]: self.get_dominant_symbol(i)
                for i in range(len(self.conditions))
            }
        }


# ═══════════════════════════════════════════════════════════════════════
# Step 2: 互資訊分析
# ═══════════════════════════════════════════════════════════════════════

class MutualInfoAnalyzer:
    """
    計算 AIM 符號與物理變數之間的互資訊，量化「哪些符號對哪個變數敏感」。

    提供三種互補指標：
    1. MI（互資訊）：整體相關強度
    2. JSD（JS 散度）：條件間分佈差異
    3. Chi²（卡方檢定）：統計顯著性
    """

    def __init__(self, codebook_size: int):
        self.codebook_size = codebook_size

    def compute_mi(self, experiment: InterventionExperiment) -> float:
        """
        計算整個實驗的 MI(variable, symbol)。

        方法：
        - 把條件標籤視為離散隨機變數 X
        - 把符號視為離散隨機變數 S
        - 計算 MI(X; S) = H(S) - H(S|X)
        """
        n_conditions = len(experiment.conditions)

        # 收集所有 (condition, symbol) 對
        joint_counts = np.zeros((n_conditions, self.codebook_size))
        for cond_idx in range(n_conditions):
            for seq in experiment.symbol_records[cond_idx]:
                for s in seq:
                    joint_counts[cond_idx, s] += 1

        total = joint_counts.sum()
        if total == 0:
            return 0.0

        joint_prob = joint_counts / total
        p_cond = joint_counts.sum(axis=1) / total    # P(X=x)
        p_sym  = joint_counts.sum(axis=0) / total    # P(S=s)

        # MI = Σ P(x,s) log[P(x,s) / (P(x)P(s))]
        mi = 0.0
        for i in range(n_conditions):
            for j in range(self.codebook_size):
                if joint_prob[i, j] > 1e-10 and p_cond[i] > 1e-10 and p_sym[j] > 1e-10:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (p_cond[i] * p_sym[j])
                    )
        return float(mi)

    def compute_per_symbol_sensitivity(
        self, experiment: InterventionExperiment
    ) -> Dict[int, float]:
        """
        計算每個符號對該變數的敏感度。

        方法：對每個符號 s，計算它在各條件下的出現率向量，
        然後計算這個向量的方差——方差越大表示 s 對條件越敏感。

        返回：{symbol_id: sensitivity_score}
        """
        sensitivity = {}
        n_conditions = len(experiment.conditions)

        for s in range(self.codebook_size):
            rates = []
            for cond_idx in range(n_conditions):
                all_symbols = []
                for seq in experiment.symbol_records[cond_idx]:
                    all_symbols.extend(seq)
                if not all_symbols:
                    rates.append(0.0)
                else:
                    rate = all_symbols.count(s) / len(all_symbols)
                    rates.append(rate)
            sensitivity[s] = float(np.var(rates))

        return sensitivity

    def compute_pairwise_jsd(
        self, experiment: InterventionExperiment
    ) -> np.ndarray:
        """
        計算所有條件兩兩之間的 JSD 矩陣。
        JSD 越大表示兩個條件下的符號分佈差異越大。
        返回：(n_conditions × n_conditions) 矩陣
        """
        n = len(experiment.conditions)
        jsd_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                p = experiment.get_symbol_distribution(i, self.codebook_size)
                q = experiment.get_symbol_distribution(j, self.codebook_size)
                jsd_val = float(jensenshannon(p, q))
                jsd_matrix[i, j] = jsd_val
                jsd_matrix[j, i] = jsd_val

        return jsd_matrix

    def chi2_significance_test(
        self, experiment: InterventionExperiment
    ) -> Tuple[float, float, bool]:
        """
        卡方檢定：符號分佈在各條件間是否有統計顯著差異。

        返回：(chi2_stat, p_value, is_significant)
        p < 0.05 表示符號分佈顯著受到該變數影響
        """
        n_conditions = len(experiment.conditions)
        observed = np.zeros((n_conditions, self.codebook_size))

        for cond_idx in range(n_conditions):
            for seq in experiment.symbol_records[cond_idx]:
                for s in seq:
                    observed[cond_idx, s] += 1

        # 移除全零列（從未出現的符號）
        col_sums = observed.sum(axis=0)
        observed = observed[:, col_sums > 0]

        if observed.shape[1] < 2:
            return 0.0, 1.0, False

        chi2, p, dof, expected = chi2_contingency(observed)
        return float(chi2), float(p), bool(p < 0.05)

    def find_condition_symbol_mapping(
        self,
        experiment: InterventionExperiment,
        sensitivity_threshold: float = 0.001,
    ) -> Dict[str, int]:
        """
        找出每個條件最具代表性的符號。
        只考慮敏感度超過閾值的符號。

        返回：{condition_label: dominant_symbol_id}
        """
        sensitivity = self.compute_per_symbol_sensitivity(experiment)
        sensitive_symbols = {
            s for s, score in sensitivity.items()
            if score >= sensitivity_threshold
        }

        mapping = {}
        for cond_idx, label in enumerate(experiment.condition_labels):
            all_symbols = []
            for seq in experiment.symbol_records[cond_idx]:
                all_symbols.extend(seq)

            # 只從敏感符號中選
            sensitive_counts = {
                s: all_symbols.count(s)
                for s in sensitive_symbols
                if s in all_symbols
            }
            if sensitive_counts:
                mapping[label] = max(sensitive_counts, key=sensitive_counts.get)
            else:
                mapping[label] = experiment.get_dominant_symbol(cond_idx)

        return mapping


# ═══════════════════════════════════════════════════════════════════════
# Step 3: 字典自動建立器
# ═══════════════════════════════════════════════════════════════════════

class AIMDictionaryBuilder:
    """
    整合干預實驗和互資訊分析，自動生成有因果證據的 AIM 字典條目。

    使用流程：
        builder = AIMDictionaryBuilder(encoder, quantizer, aim_dict)

        # 針對每個物理變數運行實驗
        exp = builder.begin_experiment("grasp_angle", [0,30,60,90], ["0°","30°","60°","90°"])
        for condition_idx, videos in enumerate(video_sets):
            for video in videos:
                builder.process_sample(exp, condition_idx, video)
        builder.finalize_experiment(exp)

        # 最終儲存
        builder.save()
    """

    def __init__(
        self,
        encoder,                    # V-JEPA 2 encoder（或任何輸出連續向量的模型）
        quantizer,                  # AIM VQ 量化器
        aim_dict: AIMDictionary,
        codebook_size: int = 64,
        num_frames: int = 16,
        device: str = "cuda",
        mi_threshold: float = 0.05,      # MI 低於此值的變數視為不相關
        sensitivity_threshold: float = 0.001,
        output_dir: str = "aim_intervention_results",
    ):
        self.encoder = encoder
        self.quantizer = quantizer
        self.aim_dict = aim_dict
        self.codebook_size = codebook_size
        self.num_frames = num_frames
        self.device = device
        self.mi_threshold = mi_threshold
        self.sensitivity_threshold = sensitivity_threshold
        self.output_dir = output_dir

        self.analyzer = MutualInfoAnalyzer(codebook_size)
        self.completed_experiments: List[Dict] = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)

    def begin_experiment(
        self,
        variable_name: str,
        conditions: List[Any],
        condition_labels: List[str],
    ) -> InterventionExperiment:
        """開始一個新的干預實驗"""
        print(f"\n[InterventionExperiment] Starting: variable='{variable_name}' "
              f"| {len(conditions)} conditions")
        return InterventionExperiment(
            variable_name, conditions, condition_labels, self.num_frames
        )

    @torch.no_grad()
    def process_sample(
        self,
        experiment: InterventionExperiment,
        condition_idx: int,
        video_tensor: torch.Tensor,      # [1, C, T, H, W] 或 [C, T, H, W]
        use_temporal_pooling: bool = True,
    ):
        """
        處理一個影片樣本：
        encoder → z → temporal pooling → quantize → record symbols
        """
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(self.device)

        # 1. Encoder 輸出連續潛向量
        z = self.encoder(video_tensor)   # [1, N_tokens, D]

        # 2. Temporal pooling（重要：不能在 patch level 量化）
        if use_temporal_pooling:
            B, N_tokens, D = z.shape
            N_spatial = N_tokens // self.num_frames
            z_frames = z.reshape(B, self.num_frames, N_spatial, D)
            z_pooled = z_frames.mean(dim=2)   # [1, T_frames, D]
        else:
            z_pooled = z.unsqueeze(1)          # 不做 pooling

        # 3. 對每一幀量化，得到符號序列
        symbols = []
        z_vectors = []
        for t in range(z_pooled.shape[1]):
            z_t = z_pooled[:, t, :]            # [1, D]
            _, indices, _ = self.quantizer(z_t, training=False)
            # indices 可能是 List（多層）或單個 tensor
            if isinstance(indices, list):
                sym = indices[0][0, 0].item()  # 取第一層的第一個 token
            else:
                sym = indices[0].item()
            symbols.append(sym)
            z_vectors.append(z_t.cpu().numpy().flatten())

        experiment.record(
            condition_idx,
            symbols,
            z_vector=np.stack(z_vectors)
        )

    def finalize_experiment(
        self,
        experiment: InterventionExperiment,
        plot: bool = True,
    ) -> Dict:
        """
        完成實驗，計算所有指標，生成字典條目，並寫入 AIMDictionary。
        """
        var_name = experiment.variable_name
        print(f"\n[Finalizing] variable='{var_name}'")
        print(f"  Samples: {experiment.summary()['samples_per_condition']}")

        # ── 計算指標 ────────────────────────────────────────────────
        mi_score = self.analyzer.compute_mi(experiment)
        chi2, p_val, is_sig = self.analyzer.chi2_significance_test(experiment)
        jsd_matrix = self.analyzer.compute_pairwise_jsd(experiment)
        sensitivity = self.analyzer.compute_per_symbol_sensitivity(experiment)
        condition_symbol_map = self.analyzer.find_condition_symbol_mapping(
            experiment, self.sensitivity_threshold
        )

        # 找出最敏感的前 N 個符號
        top_sensitive = sorted(
            sensitivity.items(), key=lambda x: x[1], reverse=True
        )[:10]

        print(f"  MI score:  {mi_score:.4f}  (threshold={self.mi_threshold})")
        print(f"  Chi² p-value: {p_val:.4f}  {'✅ significant' if is_sig else '❌ not significant'}")
        print(f"  Condition→Symbol mapping: {condition_symbol_map}")
        print(f"  Top-5 sensitive symbols: {top_sensitive[:5]}")

        # ── 決定是否寫入字典 ────────────────────────────────────────
        if mi_score < self.mi_threshold:
            print(f"  ⚠️  MI too low ({mi_score:.4f}), skipping dictionary entry")
            result = {"variable": var_name, "mi": mi_score, "written": False}
            self.completed_experiments.append(result)
            return result

        if not is_sig:
            print(f"  ⚠️  Not statistically significant (p={p_val:.4f}), "
                  f"writing with caution flag")

        # ── 為每個條件寫入字典條目 ──────────────────────────────────
        mean_jsd = float(np.mean(jsd_matrix[jsd_matrix > 0])) if jsd_matrix.any() else 0.0

        for cond_idx, label in enumerate(experiment.condition_labels):
            dominant_sym = condition_symbol_map.get(label, -1)
            if dominant_sym < 0:
                continue

            aim_id_str = str([dominant_sym])   # AIMDictionary 期望字串格式

            evidence = {
                "experiment": f"intervention_{var_name}",
                "MI_score": round(mi_score, 4),
                "chi2_p_value": round(p_val, 4),
                "is_significant": is_sig,
                "mean_JSD_across_conditions": round(mean_jsd, 4),
                "condition_value": str(experiment.conditions[cond_idx]),
                "conditions_tested": experiment.condition_labels,
                "symbol_mapping": {
                    lbl: condition_symbol_map.get(lbl, -1)
                    for lbl in experiment.condition_labels
                },
                "top_sensitive_symbols": [
                    {"symbol": s, "sensitivity": round(sc, 6)}
                    for s, sc in top_sensitive[:5]
                ],
                "samples_this_condition": len(
                    experiment.symbol_records[cond_idx]),
                "timestamp": datetime.now().isoformat(),
            }

            # human_label 格式：variable_name=condition_label
            human_label = f"{var_name}={label}"

            # context 格式對齊 AIMDictionary 的 key 結構
            # AIMDictionary 用 (aim_id, human_label, context) 作為唯一 key
            # context 必須是 hashable（字串）
            context = json.dumps(evidence, sort_keys=True)

            self.aim_dict.add_entry(aim_id_str, human_label, context)
            print(f"  ✅ Written: {aim_id_str} → '{human_label}'")

        # ── 視覺化 ──────────────────────────────────────────────────
        if plot:
            self._plot_experiment(experiment, jsd_matrix, sensitivity, var_name)

        result = {
            "variable": var_name,
            "mi": mi_score,
            "p_value": p_val,
            "is_significant": is_sig,
            "condition_symbol_map": condition_symbol_map,
            "written": True,
        }
        self.completed_experiments.append(result)
        return result

    def _plot_experiment(
        self,
        experiment: InterventionExperiment,
        jsd_matrix: np.ndarray,
        sensitivity: Dict[int, float],
        var_name: str,
    ):
        """生成三張診斷圖"""
        n_conds = len(experiment.conditions)
        labels = experiment.condition_labels

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Intervention Analysis: {var_name}", fontsize=13, fontweight="bold"
        )

        # 圖 1：各條件的符號分佈（前 20 個符號）
        ax = axes[0]
        top_syms = sorted(sensitivity, key=sensitivity.get, reverse=True)[:20]
        x = np.arange(len(top_syms))
        width = 0.8 / n_conds
        colors = plt.cm.Set2(np.linspace(0, 1, n_conds))

        for ci, (label, color) in enumerate(zip(labels, colors)):
            dist = experiment.get_symbol_distribution(ci, self.codebook_size)
            heights = [dist[s] for s in top_syms]
            ax.bar(x + ci * width, heights, width, label=label, color=color, alpha=0.8)

        ax.set_xticks(x + width * (n_conds - 1) / 2)
        ax.set_xticklabels([str(s) for s in top_syms], rotation=45, fontsize=8)
        ax.set_xlabel("Symbol ID (top-20 by sensitivity)")
        ax.set_ylabel("Frequency")
        ax.set_title("Symbol Distribution per Condition")
        ax.legend(fontsize=8)

        # 圖 2：JSD 熱力圖（條件兩兩之間的分佈差異）
        ax = axes[1]
        im = ax.imshow(jsd_matrix, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_conds))
        ax.set_yticks(range(n_conds))
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        plt.colorbar(im, ax=ax)
        for i in range(n_conds):
            for j in range(n_conds):
                ax.text(j, i, f"{jsd_matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if jsd_matrix[i, j] > 0.5 else "black")
        ax.set_title("Pairwise JSD Between Conditions")

        # 圖 3：符號敏感度排名（前 30 個符號）
        ax = axes[2]
        top30 = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:30]
        syms, scores = zip(*top30) if top30 else ([], [])
        bar_colors = ["#e74c3c" if sc > np.mean(scores) * 2 else "#3498db"
                      for sc in scores]
        ax.bar(range(len(syms)), scores, color=bar_colors)
        ax.set_xticks(range(len(syms)))
        ax.set_xticklabels([str(s) for s in syms], rotation=45, fontsize=8)
        ax.set_xlabel("Symbol ID")
        ax.set_ylabel("Sensitivity (variance across conditions)")
        ax.set_title("Symbol Sensitivity Ranking")

        plt.tight_layout()
        fig_path = f"{self.output_dir}/figures/intervention_{var_name}.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  📊 Plot saved: {fig_path}")

    def save(self):
        """將所有實驗結果寫入 AIMDictionary 並儲存"""
        self.aim_dict.save()
        summary_path = f"{self.output_dir}/experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.completed_experiments, f, indent=2)
        print(f"\n✅ AIM Dictionary saved.")
        print(f"✅ Experiment summary: {summary_path}")
        print(f"   Total experiments: {len(self.completed_experiments)}")
        written = sum(1 for e in self.completed_experiments if e.get("written"))
        print(f"   Written to dictionary: {written}")


# ═══════════════════════════════════════════════════════════════════════
# Step 4: 交叉驗證器
# ═══════════════════════════════════════════════════════════════════════

class CrossValidator:
    """
    用未見過的新場景驗證字典的泛化能力。

    給定一個新的條件值（如 45° 傾角），
    預測它的符號應該落在哪個範圍，
    然後實際量化後檢查是否符合預測。
    """

    def __init__(
        self,
        builder: AIMDictionaryBuilder,
        codebook_size: int = 64,
    ):
        self.builder = builder
        self.codebook_size = codebook_size
        self.results: List[Dict] = []

    def validate(
        self,
        variable_name: str,
        new_condition_label: str,
        new_condition_value: Any,
        video_tensor: torch.Tensor,
        expected_symbol_range: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        驗證新場景的符號是否符合字典的預測。

        expected_symbol_range：如果已知應落在 [s_min, s_max] 之間，可以指定。
        若不指定，則純粹記錄觀察到的符號。
        """
        # 建立臨時實驗收集符號
        temp_exp = InterventionExperiment(
            variable_name, [new_condition_value], [new_condition_label]
        )
        self.builder.process_sample(temp_exp, 0, video_tensor)
        observed_symbols = []
        for seq in temp_exp.symbol_records[0]:
            observed_symbols.extend(seq)

        dominant = temp_exp.get_dominant_symbol(0)
        dist = temp_exp.get_symbol_distribution(0, self.codebook_size)

        # 從字典中查找同一變數的已知條目
        known_entries = self.builder.aim_dict.get_entries()
        related = [
            e for e in known_entries
            if e["human_label"].startswith(f"{variable_name}=")
        ]

        # 計算新觀察與已知條目的 JSD
        known_symbols = {}
        for entry in related:
            lbl = entry["human_label"].split("=", 1)[1]
            aim_id = int(json.loads(entry["aim_id"])[0])
            known_symbols[lbl] = aim_id

        # 判斷是否通過驗證
        passed = None
        if expected_symbol_range is not None:
            passed = expected_symbol_range[0] <= dominant <= expected_symbol_range[1]

        result = {
            "variable": variable_name,
            "new_condition": new_condition_label,
            "observed_dominant_symbol": dominant,
            "observed_distribution_entropy": float(
                -np.sum(dist * np.log(dist + 1e-10))
            ),
            "known_symbol_mapping": known_symbols,
            "expected_range": expected_symbol_range,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        }

        status = "✅ PASS" if passed else ("❌ FAIL" if passed is False else "📋 RECORDED")
        print(f"\n[CrossValidation] {variable_name}={new_condition_label}")
        print(f"  Dominant symbol: {dominant}")
        print(f"  Known mapping:   {known_symbols}")
        print(f"  Result: {status}")

        self.results.append(result)
        return result

    def report(self) -> Dict:
        """輸出驗證總報告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"] is True)
        failed = sum(1 for r in self.results if r["passed"] is False)
        recorded = total - passed - failed

        report = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "recorded_only": recorded,
            "pass_rate": passed / total if total > 0 else 0.0,
            "details": self.results,
        }
        print(f"\n[CrossValidation Report]")
        print(f"  Total: {total} | Pass: {passed} | Fail: {failed} | "
              f"Pass rate: {report['pass_rate']:.1%}")
        return report


# ═══════════════════════════════════════════════════════════════════════
# 使用範例（不需要真實模型也能理解流程）
# ═══════════════════════════════════════════════════════════════════════

def example_usage():
    """
    展示完整的使用流程。
    在真實環境中，把 MockEncoder/MockQuantizer 換成真實的 V-JEPA 2 元件。
    """

    # ── 模擬元件（測試用） ──────────────────────────────────────────
    class MockEncoder(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, 16 * 14 * 14, 1408)  # V-JEPA ViT-g 格式

    class MockQuantizer(torch.nn.Module):
        def forward(self, z, training=False):
            B, D = z.shape
            indices = torch.randint(0, 64, (B, 1))
            return z, [indices], torch.tensor(0.0)

    # ── 初始化 ──────────────────────────────────────────────────────
    encoder   = MockEncoder()
    quantizer = MockQuantizer()
    aim_dict  = AIMDictionary("aim_dictionary_intervention.json")
    builder   = AIMDictionaryBuilder(
        encoder, quantizer, aim_dict,
        codebook_size=64, num_frames=16
    )

    # ── 實驗 1：抓握角度 ────────────────────────────────────────────
    exp_angle = builder.begin_experiment(
        variable_name="grasp_angle",
        conditions=[0, 30, 60, 90],
        condition_labels=["0deg", "30deg", "60deg", "90deg"],
    )

    # 每個條件提供 20 個樣本
    for cond_idx in range(4):
        for _ in range(20):
            dummy_video = torch.randn(1, 3, 16, 224, 224)
            builder.process_sample(exp_angle, cond_idx, dummy_video)

    builder.finalize_experiment(exp_angle)

    # ── 實驗 2：物體重量 ────────────────────────────────────────────
    exp_weight = builder.begin_experiment(
        variable_name="object_weight",
        conditions=[0.1, 0.5, 1.0, 2.0],
        condition_labels=["light", "medium", "heavy", "very_heavy"],
    )
    for cond_idx in range(4):
        for _ in range(20):
            dummy_video = torch.randn(1, 3, 16, 224, 224)
            builder.process_sample(exp_weight, cond_idx, dummy_video)

    builder.finalize_experiment(exp_weight)

    # ── 儲存 ────────────────────────────────────────────────────────
    builder.save()

    # ── 交叉驗證 ────────────────────────────────────────────────────
    validator = CrossValidator(builder, codebook_size=64)
    dummy_new_video = torch.randn(1, 3, 16, 224, 224)
    validator.validate(
        variable_name="grasp_angle",
        new_condition_label="45deg",
        new_condition_value=45,
        video_tensor=dummy_new_video,
        expected_symbol_range=(10, 50),  # 預期落在這個範圍
    )
    validator.report()


if __name__ == "__main__":
    example_usage()
