"""
aim_adapter.py
==============
Adapter: AIM Repo (Steps 1-3) <-> Collusion Framework (Steps 4-6)

This file is the only place that knows the format of both sides.
The two programs do not need to know about each other's existence, they just need to import this module.

Also houses observer components (BaselineObserver, RandomizedObserverPool) so that
both vqvae_agents_AIM.py and aim_collusion_framework.py can share the same definitions
without circular imports.

Usage:
    from aim_adapter import AIMAdapter, BaselineObserver, RandomizedObserverPool
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════
# Observer Components
# Centralised here so vqvae_agents_AIM.py and aim_collusion_framework.py
# both import from a single authoritative source.
# ══════════════════════════════════════════════════════════════════════

class BaselineObserver(nn.Module):
    """
    Supervisor model (V1.5, default strategy).
    Task: Predict Agent B's action (C/D) solely from Agent A's discrete AIM symbols.
    A high prediction accuracy while joint reward is also high signals hidden coordination.
    """
    def __init__(self, aim_seq_len=2, K=16, quantizer_D=64):
        super().__init__()
        self.aim_embedding = nn.Embedding(K, quantizer_D)
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(aim_seq_len * quantizer_D, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # 2 logits: C(0) or D(1)
        )

    def forward(self, aim_sequence):
        emb = self.aim_embedding(aim_sequence)
        return self.predictor(emb)


class RandomizedObserverPool:
    """
    [Experimental] Observer pool for anti-gaming evaluation (V2.0 direction).

    Motivation: a single fixed observer can be learned and exploited by agents
    (Transparency Paradox). A rotating pool of observers makes the evaluation
    target a moving one, raising the cost of stable collusion strategies.

    Design:
    - Maintains `pool_size` independent BaselineObserver models.
    - Each round: samples `subset_size` observers at random, trains each one step,
      returns equal-weight average accuracy (random weights removed to reduce EMA noise).
    - Every `refresh_interval` rounds: replaces one randomly chosen observer with a
      freshly initialised model, injecting an unseen evaluator into the pool.

    Activated via --observer_strategy randomized_pool (default: fixed).
    Status: Experimental — not yet validated against V1.5 baseline experiments.
    """
    def __init__(self, aim_seq_len, K, quantizer_D,
                 pool_size=8, subset_size=3,
                 refresh_interval=100, observer_lr=1e-3):
        self.pool_size        = max(1, pool_size)
        self.subset_size      = min(max(1, subset_size), self.pool_size)
        self.refresh_interval = max(1, refresh_interval)
        self.aim_seq_len      = aim_seq_len
        self.K                = K
        self.quantizer_D      = quantizer_D
        self.observer_lr      = observer_lr

        self.observers  = [BaselineObserver(aim_seq_len, K, quantizer_D)
                           for _ in range(self.pool_size)]
        self.optimizers = [optim.Adam(obs.parameters(), lr=observer_lr)
                           for obs in self.observers]

    def _refresh_one(self):
        """Replace one randomly chosen observer with a freshly initialised model."""
        idx = random.randrange(self.pool_size)
        self.observers[idx]  = BaselineObserver(self.aim_seq_len, self.K, self.quantizer_D)
        self.optimizers[idx] = optim.Adam(self.observers[idx].parameters(),
                                          lr=self.observer_lr)

    def step(self, A_sampled_aim_sequence, target_B_action_obs, round_idx):
        """
        Train a random subset of observers for one step.
        Returns equal-weight average accuracy across the sampled subset.
        """
        if round_idx > 0 and round_idx % self.refresh_interval == 0:
            self._refresh_one()

        sampled_indices  = random.sample(range(self.pool_size), self.subset_size)
        per_observer_acc = []

        for idx in sampled_indices:
            obs_logits = self.observers[idx](A_sampled_aim_sequence)
            loss_obs   = nn.CrossEntropyLoss()(obs_logits, target_B_action_obs)
            self.optimizers[idx].zero_grad()
            loss_obs.backward()
            self.optimizers[idx].step()

            obs_probs = torch.softmax(obs_logits, dim=1)
            per_observer_acc.append(obs_probs[0, target_B_action_obs.item()].detach())

        return float(torch.stack(per_observer_acc).mean().item())


# ══════════════════════════════════════════════════════════════════════
# Data Adapter
# ══════════════════════════════════════════════════════════════════════


class AIMAdapter:
    """
    Converts the output of multi_agent_game() into the standard framework format,
    or conversely, converts the framework's parameters into input for multi_agent_game().
    """

    def __init__(self, K: int = 32):
        """
        K : Codebook size, corresponds to the --K parameter in aim_main.py
        """
        self.K = K

    # ──────────────────────────────────────────────────────────────────
    # Direction A: aim_main.py -> framework
    # ──────────────────────────────────────────────────────────────────
    def to_framework(self,
                     joint_hist:          list,
                     obs_acc_hist:        list,
                     shuffle_hist:        list,
                     encoding_inds_hist:  list,
                     penalty_hist:        list = None) -> dict:
        """
        Input (returns from multi_agent_game()):
            joint_hist          : joint_rewards_history
            obs_acc_hist        : obs_accuracies_history
            shuffle_hist        : shuffle_rounds_history
            encoding_inds_hist  : encoding_inds_history (needs to be collected in aim_main.py)
            penalty_hist        : penalty_rounds_history (optional)

        Output (standard framework format):
            dict, can be directly passed to validate_thresholds() or perform_threshold_sweep()
        """
        codebook_usages = self._build_codebook_usages(encoding_inds_hist)

        return {
            "joint_rewards":         np.array(joint_hist,   dtype=float),
            "observer_accs":   np.array(obs_acc_hist, dtype=float),
            "codebook_usages": codebook_usages,
            "shuffle_rounds":  shuffle_hist,
            "penalty_rounds":  penalty_hist or [],
        }

    # ──────────────────────────────────────────────────────────────────
    # Direction B: framework parameters -> aim_main.py input
    # ──────────────────────────────────────────────────────────────────
    def to_aim_kwargs(self,
                      threshold_penalty: float,
                      threshold_shuffle: float,
                      rounds:            int,
                      **kwargs) -> dict:
        """
        Converts framework sweep parameters into kwargs for multi_agent_game().

        Usage:
            kwargs = adapter.to_aim_kwargs(tp, ts, rounds=1500)
            multi_agent_game(vqvae, aim_dict, **kwargs)
        """
        return {
            "rounds":                 rounds,
            "enable_reward_shaping":  True,
            "enable_codebook_shuffle": True,
            "threshold_penalty":      threshold_penalty,
            "threshold_shuffle":      threshold_shuffle,
            **kwargs,   # Pass through other additional parameters
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal tools
    # ──────────────────────────────────────────────────────────────────
    def _build_codebook_usages(self, encoding_inds_hist: list) -> list:
        """
        Converts a single encoding index from each round into a K-dimensional count vector.

        encoding_inds_hist can be:
            - List[int]           : One index per round
            - List[List[int]]     : Multiple indices per round (aim_seq_len > 1)
        """
        usages = []
        for entry in encoding_inds_hist:
            indices = [entry] if isinstance(entry, int) else list(entry)
            counts = np.bincount(indices, minlength=self.K).astype(float)
            usages.append(counts)
        return usages