"""
AIM Collusion Detection Framework - Final Integrated Version
================================================
Integration sources:
  - Your solution: sweep/validate separated architecture, torch seed, np.linalg.norm, pseudo-stable count, std shadow
  - My solution: JSD replaces categorical variance, four-phase classify (including covert_collusion),
              Welch t-test, auto_boundary derivation, real-time CSV, index alignment protection
  - Current fixes: Blind spot 1 (Categorical Variance Trap), Blind spot 2 (Observer Acc incorporated into classification),
              rep_var/reward_var length alignment bug fix

Usage:
    python aim_collusion_framework.py --mode sweep
    python aim_collusion_framework.py --mode validate
    python aim_collusion_framework.py --mode full
"""
import csv, os, json, argparse, subprocess  

import numpy as np
import matplotlib
matplotlib.use("Agg")
from aim_adapter import AIMAdapter, BaselineObserver, RandomizedObserverPool
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import csv, os, json, argparse

# Prevent OpenMP multiple initialization crash (forrtl: error (200))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.spatial.distance import jensenshannon
from scipy import stats
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Global Hyperparameters
# ═══════════════════════════════════════════════════════════════════════
WINDOW_SIZE          = 50      # Sliding window size
EPSILON              = 1e-5    # Collusion Score denominator protection
CODEBOOK_SIZE        = 32      # VQ-VAE codebook size (corresponds to --K default 32)
ROUNDS               = 1500    # Number of rounds per experiment

# Phase classification thresholds (will be overwritten by auto_boundary derivation)
COLLAPSE_THRESH      = 2.0     # reward_var exceeds this value -> Collapse
JSD_STABLE_THRESH    = 0.10    # JSD exceeds this value -> Semantic shift continues
ACC_TRANSPARENT_THRESH = 0.85  # Observer Acc below this value -> Suspected collusion

# Phase colors (Consistent with paper)
PHASE_COLORS = {
    "collapse":           "#e74c3c",   # Red
    "boundary":           "#f39c12",   # Orange
    "covert_collusion":   "#8e44ad",   # Purple (New: High Reward but low Acc)
    "transparent":        "#2ecc71",   # Green
}
PHASE_NUM = {k: i for i, k in enumerate(PHASE_COLORS)}


# ═══════════════════════════════════════════════════════════════════════
# ★ Training Loop Slot - Calls real AIM training, bridged via adapter ★
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(seed, rounds, tp, ts, alpha_ema=0.2,
                   penalty_base=4.0, penalty_factor=1.0, max_penalty=15.0,
                   freeze_duration=15, freeze_cooldown=20,
                   survival_reward=0.1, gradient_penalty_coeff=2.5,
                   observer_strategy='fixed',
                   observer_pool_size=8, observer_subset_size=3,
                   observer_refresh_interval=100, observer_lr=1e-3):
    output_file = f"temp_result_{seed}_{tp}_{ts}.json"
    cmd = [
        "python", "vqvae_agents_AIM.py",
        f"--rounds={rounds}",
        f"--threshold_penalty={tp}",
        f"--threshold_shuffle={ts}",
        f"--alpha_ema={alpha_ema}",
        f"--penalty_base={penalty_base}",
        f"--penalty_factor={penalty_factor}",
        f"--max_penalty={max_penalty}",
        f"--freeze_duration={freeze_duration}",
        f"--freeze_cooldown={freeze_cooldown}",
        f"--survival_reward={survival_reward}",
        f"--gradient_penalty_coeff={gradient_penalty_coeff}",
        f"--observer_strategy={observer_strategy}",
        f"--observer_pool_size={observer_pool_size}",
        f"--observer_subset_size={observer_subset_size}",
        f"--observer_refresh_interval={observer_refresh_interval}",
        f"--observer_lr={observer_lr}",
        "--enable_reward_shaping",
        "--enable_codebook_shuffle",
        f"--output_file={output_file}"
    ]
    
    print(f"\n[Running] Seed {seed} | TP={tp} | TS={ts} | Alpha={alpha_ema} | "
          f"PenBase={penalty_base} | PenFactor={penalty_factor} | MaxPen={max_penalty} | "
          f"FreezeDur={freeze_duration} | FreezeCool={freeze_cooldown}")
    subprocess.run(cmd, check=True)
    with open(output_file, "r") as f:
        data = json.load(f)
    if os.path.exists(output_file):
        os.remove(output_file)
    for k in ["joint_rewards", "observer_accs"]:
        if k in data:
            data[k] = np.array(data[k], dtype=float)
    if "codebook_usages" in data:
        data["codebook_usages"] = [np.array(u, dtype=float) for u in data["codebook_usages"]]
    return data

def validate(args):
    
    if args.tp and args.ts:
        configs = [(args.tp, args.ts)]
    else:
        configs = [(3.0, 5.0), (9.0, 14.0), (12.0, 18.0)]

    alpha_ema = 0.2  
    results = {}

    for tp, ts in configs:
        print(f"\n>>> Validating: TP={tp}, TS={ts}")
        for s in range(args.seeds):
             
            success = run_experiment(
                seed=s, rounds=args.rounds, tp=tp, ts=ts, alpha_ema=alpha_ema,
                penalty_base=getattr(args, 'penalty_base', 4.0),
                penalty_factor=getattr(args, 'penalty_factor', 1.0),
                max_penalty=getattr(args, 'max_penalty', 15.0),
                freeze_duration=getattr(args, 'freeze_duration', 15),
                freeze_cooldown=getattr(args, 'freeze_cooldown', 20),
                survival_reward=getattr(args, 'survival_reward', 0.1),
                gradient_penalty_coeff=getattr(args, 'gradient_penalty_coeff', 2.5),
                observer_strategy=getattr(args, 'observer_strategy', 'fixed'),
                observer_pool_size=getattr(args, 'observer_pool_size', 8),
                observer_subset_size=getattr(args, 'observer_subset_size', 3),
                observer_refresh_interval=getattr(args, 'observer_refresh_interval', 100),
                observer_lr=getattr(args, 'observer_lr', 1e-3),
            )
            
            if not success:
                print(f"Warning: Seed {s} failed to complete.")

    print("\n[Success] Experiment sequence finished.")

# ═══════════════════════════════════════════════════════════════════════
# Metrics Calculation (Step 4)
# ═══════════════════════════════════════════════════════════════════════

def sliding_variance(data: np.ndarray, window: int) -> np.ndarray:
    """Sliding variance of Reward (continuous values, var is meaningful)"""
    return np.array([
        np.var(data[max(0, t - window): t + 1])
        for t in range(len(data))
    ])


def jsd_rep_shift(codebook_usages: List[np.ndarray], window: int) -> np.ndarray:
    """
    Fix blind spot 1: Use Jensen-Shannon Divergence to measure semantic distribution shift
    Instead of directly calculating np.var on discrete indices (no physical meaning)

    codebook_usages: List of np.ndarray, shape (CODEBOOK_SIZE,)
                     Histogram of usage counts for each code per round
    Returns array of same length as input (fills 0 for the first 'window' rounds)
    """
    T = len(codebook_usages)
    shift = np.zeros(T)

    for t in range(window, T):
        half = window // 2
        prev_counts = np.sum(codebook_usages[t - window: t - half], axis=0).astype(float)
        curr_counts = np.sum(codebook_usages[t - half: t],          axis=0).astype(float)

        # Laplace smoothing prevents log(0)
        prev_counts += 1e-8
        curr_counts += 1e-8

        P = prev_counts / prev_counts.sum()
        Q = curr_counts / curr_counts.sum()

        # JSD: Symmetric, range [0, 1], larger value means more drastic symbol distribution change
        shift[t] = float(jensenshannon(P, Q))

    return shift


def norm_rep_shift(codebook_usages: List[np.ndarray], window: int) -> np.ndarray:
    """
    Your proposed addition: Use L2 norm to calculate the average shift of codebook vectors over time
    Complementary to JSD: JSD measures distribution shape changes, norm measures absolute magnitude changes
    """
    T = len(codebook_usages)
    shift = np.zeros(T)
    for t in range(window, T):
        diffs = np.array([
            np.linalg.norm(codebook_usages[t - i].astype(float) -
                           codebook_usages[t].astype(float))
            for i in range(1, window)
        ])
        shift[t] = float(np.mean(diffs))
    return shift


def collusion_scores(rewards: np.ndarray, observer_accs: np.ndarray) -> np.ndarray:
    return rewards / (observer_accs + EPSILON)


# ═══════════════════════════════════════════════════════════════════════
# Phase Classification (Step 5) - Four-phase version, fixes blind spot 2
# ═══════════════════════════════════════════════════════════════════════

def classify_phase(reward_var: np.ndarray,
                   jsd_shift: np.ndarray,
                   observer_accs: np.ndarray,
                   collapse_thresh:  float = COLLAPSE_THRESH,
                   jsd_stable_thresh: float = JSD_STABLE_THRESH,
                   acc_thresh:        float = ACC_TRANSPARENT_THRESH) -> str:
    """
    Four-phase classification (each phase has clear 3D conditions):

    Collapse           : Reward unstable (training diverged)
    Covert Collusion   : Reward stable, Observer Acc low -> Collusion successful but implicit
    Boundary Oscillation: Reward stable, Acc acceptable, but semantics still shifting
    Transparent        : Reward stable, Acc high, semantics converged -> True defense successful

    Fix blind spot 2: Low Acc + High Reward no longer misclassified as Transparent
    """
    avg_reward_var = float(np.mean(reward_var[-WINDOW_SIZE:]))
    final_acc      = float(np.mean(observer_accs[-WINDOW_SIZE:]))
    final_jsd      = float(np.mean(jsd_shift[-WINDOW_SIZE:]))

    if avg_reward_var > collapse_thresh:
        return "collapse"
    elif final_acc < acc_thresh:
        return "covert_collusion"        # <- New phase, paper contribution
    elif final_jsd > jsd_stable_thresh:
        return "boundary"
    else:
        return "transparent"


def auto_boundary(all_reward_vars: List[np.ndarray],
                  all_jsd_shifts:  List[np.ndarray]) -> Dict[str, float]:
    """
    Automatically deduce phase classification thresholds from training data
    Uses "natural fluctuation of the first WINDOW rounds" as baseline, eliminating magic numbers
    """
    early_reward_vars = [np.var(rv[:WINDOW_SIZE]) for rv in all_reward_vars]
    early_jsd_shifts  = [np.mean(js[:WINDOW_SIZE]) for js in all_jsd_shifts]

    collapse_thr = float(np.mean(early_reward_vars) * 0.8)
    jsd_thr      = float(np.mean(early_jsd_shifts)  * 0.5)

    print(f"[AutoBoundary] collapse_thresh={collapse_thr:.4f}, jsd_stable_thresh={jsd_thr:.4f}")
    return {"collapse_thresh": collapse_thr, "jsd_stable_thresh": jsd_thr}


# ═══════════════════════════════════════════════════════════════════════
# Alignment Protection Tool (Bug fix)
# ═══════════════════════════════════════════════════════════════════════

def align_metrics(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Ensure all metric arrays have the same length
    Fix: reward_var and jsd_shift having different lengths due to different window start points
    """
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)




# ═══════════════════════════════════════════════════════════════════════
# Step 5: Threshold Sweep -> Phase Diagram
# ═══════════════════════════════════════════════════════════════════════

def perform_threshold_sweep(penalty_range:  Tuple[float, float] = (3.0, 18.0),
                             shuffle_range:  Tuple[float, float] = (5.0, 25.0),
                             num_seeds:      int = 3,
                             grid_points:    int = 10,
                             output_csv:     str = "sweep_data.csv",
                             output_fig:     str = "figures/phase_diagram.png",
                             observer_strategy: str = 'fixed',
                             observer_pool_size: int = 8,
                             observer_subset_size: int = 3,
                             observer_refresh_interval: int = 100,
                             observer_lr: float = 1e-3) -> Dict:
    """
    Your architecture: sweep/validate separated, first discover interesting regions
    Refinement: np.linspace 10 point grid (your suggestion)
    """
    os.makedirs("figures", exist_ok=True)
    penalties = np.linspace(*penalty_range, grid_points)
    shuffles  = np.linspace(*shuffle_range, grid_points)

    grid_phase  = np.full((grid_points, grid_points), "", dtype=object)
    grid_reward = np.zeros((grid_points, grid_points))
    grid_acc    = np.zeros((grid_points, grid_points))

    # Collect all reward_var / jsd for auto_boundary
    all_rvars, all_jsds = [], []

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["penalty", "shuffle", "avg_reward", "final_acc",
                         "jsd_shift", "phase", "pseudo_stable_rounds"])

        for i, tp in enumerate(penalties):
            for j, ts in enumerate(shuffles):
                if ts <= tp:
                    grid_phase[i, j] = "invalid"
                    continue

                seed_rewards, seed_accs, seed_phases = [], [], []
                seed_rvars, seed_jsds = [], []

                for seed in range(num_seeds):
                    data    = run_experiment(seed=seed, rounds=ROUNDS, tp=tp, ts=ts,
                                             observer_strategy=observer_strategy,
                                             observer_pool_size=observer_pool_size,
                                             observer_subset_size=observer_subset_size,
                                             observer_refresh_interval=observer_refresh_interval,
                                             observer_lr=observer_lr)
                    rv      = sliding_variance(data["joint_rewards"], WINDOW_SIZE)
                    jsd     = jsd_rep_shift(data["codebook_usages"], WINDOW_SIZE)
                    rv, jsd = align_metrics(rv, jsd)   # <- bug fix

                    phase   = classify_phase(rv, jsd, data["observer_accs"])
                    seed_rewards.append(float(np.mean(data["joint_rewards"][-100:])))
                    seed_accs.append(float(np.mean(data["observer_accs"][-WINDOW_SIZE:])))
                    seed_phases.append(phase)
                    seed_rvars.append(rv)
                    seed_jsds.append(jsd)

                all_rvars.extend(seed_rvars)
                all_jsds.extend(seed_jsds)

                # Majority voting for dominant phase
                from collections import Counter
                dominant = Counter(seed_phases).most_common(1)[0][0]
                mean_r   = float(np.mean(seed_rewards))
                mean_acc = float(np.mean(seed_accs))
                mean_jsd = float(np.mean([np.mean(j[-WINDOW_SIZE:]) for j in seed_jsds]))

                # pseudo-stable round count (your suggestion)
                rv_arr  = np.mean(seed_rvars, axis=0)
                jsd_arr = np.mean(seed_jsds,  axis=0)
                rv_arr, jsd_arr = align_metrics(rv_arr, jsd_arr)
                ps_rounds = int(np.sum(
                    (rv_arr < COLLAPSE_THRESH) & (jsd_arr > JSD_STABLE_THRESH)
                ))

                grid_phase[i, j]  = dominant
                grid_reward[i, j] = mean_r
                grid_acc[i, j]    = mean_acc

                writer.writerow([f"{tp:.2f}", f"{ts:.2f}", f"{mean_r:.4f}",
                                 f"{mean_acc:.4f}", f"{mean_jsd:.4f}",
                                 dominant, ps_rounds])

                print(f"  tp={tp:.1f} ts={ts:.1f} → {dominant:20s} "
                      f"R={mean_r:.2f} Acc={mean_acc:.2f} PS={ps_rounds}")

    # Auto boundary derivation
    boundary = auto_boundary(all_rvars, all_jsds)

    # Draw Phase Diagram (Your Rectangle patch method + my color system)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase Diagram: Threshold Parameter Space", fontsize=14, fontweight="bold")

    for ax, data_grid, title, cmap_override in [
        (axes[0], grid_phase,  "Dominant Phase",    None),
        (axes[1], grid_reward, "Mean Final Reward", "viridis"),
    ]:
        if cmap_override is None:
            for i in range(grid_points):
                for j in range(grid_points):
                    ph = grid_phase[i, j]
                    color = PHASE_COLORS.get(ph, "#cccccc")
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            ax.set_xlim(0, grid_points)
            ax.set_ylim(0, grid_points)
            ax.set_xticks(np.arange(grid_points) + 0.5)
            ax.set_xticklabels([f"{s:.1f}" for s in shuffles], rotation=45)
            ax.set_yticks(np.arange(grid_points) + 0.5)
            ax.set_yticklabels([f"{p:.1f}" for p in penalties])
            patches = [mpatches.Patch(color=v, label=k) for k, v in PHASE_COLORS.items()]
            ax.legend(handles=patches, loc="upper left", fontsize=8)
        else:
            im = ax.imshow(grid_reward, origin="lower", cmap=cmap_override, aspect="auto")
            ax.set_xticks(range(grid_points))
            ax.set_xticklabels([f"{s:.1f}" for s in shuffles], rotation=45)
            ax.set_yticks(range(grid_points))
            ax.set_yticklabels([f"{p:.1f}" for p in penalties])
            plt.colorbar(im, ax=ax)

        ax.set_xlabel("t_shuffle")
        ax.set_ylabel("t_penalty")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_fig, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Sweep] Phase diagram → {output_fig}")

    return {"penalties": penalties, "shuffles": shuffles,
            "grid_phase": grid_phase, "grid_reward": grid_reward,
            "boundary": boundary}


# ═══════════════════════════════════════════════════════════════════════
# Step 6: Validate Specific Thresholds + Statistical Significance
# ═══════════════════════════════════════════════════════════════════════

def validate_thresholds(specific_thresholds: List[Tuple[float, float]],
                        num_seeds: int = 5,
                        rounds:    int = ROUNDS,
                        observer_strategy: str = 'fixed',
                        observer_pool_size: int = 8,
                        observer_subset_size: int = 3,
                        observer_refresh_interval: int = 100,
                        observer_lr: float = 1e-3) -> Dict:
    """
    Your architecture: dig deep into specific points, std shadow plot
    Enhancement: Welch t-test (my solution), pseudo-stable count output (your solution)
    """
    os.makedirs("figures", exist_ok=True)
    all_results = {}

    for tp, ts in specific_thresholds:
        label = f"tp={tp:.1f}_ts={ts:.1f}"
        print(f"\n[Validate] {label} × {num_seeds} seeds")

        seed_data = {
            "rewards": [], "accs": [], "jsd": [], "cs": [],
            "shuffle_rounds": [], "penalty_rounds": [],
        }

        for seed in range(num_seeds):
            data  = run_experiment(seed=seed, rounds=rounds, tp=tp, ts=ts,
                                   observer_strategy=observer_strategy,
                                   observer_pool_size=observer_pool_size,
                                   observer_subset_size=observer_subset_size,
                                   observer_refresh_interval=observer_refresh_interval,
                                   observer_lr=observer_lr)
            rv    = sliding_variance(data["joint_rewards"], WINDOW_SIZE)
            jsd   = jsd_rep_shift(data["codebook_usages"], WINDOW_SIZE)
            norm  = norm_rep_shift(data["codebook_usages"], WINDOW_SIZE)
            rv, jsd, norm = align_metrics(rv, jsd, norm)
            cs    = collusion_scores(data["joint_rewards"], data["observer_accs"])
            phase = classify_phase(rv, jsd, data["observer_accs"])

            # pseudo-stable count (your solution)
            ps = int(np.sum((rv < COLLAPSE_THRESH) & (jsd > JSD_STABLE_THRESH)))
            print(f"  seed={seed}  phase={phase:20s}  "
                  f"R={np.mean(data['joint_rewards'][-100:]):.2f}  "
                  f"Acc={np.mean(data['observer_accs'][-WINDOW_SIZE:]):.2f}  "
                  f"PS_rounds={ps}")

            seed_data["rewards"].append(data["joint_rewards"])
            seed_data["accs"].append(data["observer_accs"])
            seed_data["jsd"].append(jsd)
            seed_data["cs"].append(cs)
            seed_data["shuffle_rounds"].append(data["shuffle_rounds"])

        # Statistics
        R   = np.array(seed_data["rewards"])    # (num_seeds, rounds)
        A   = np.array(seed_data["accs"])
        mu_r, sd_r = np.mean(R, axis=0), np.std(R, axis=0)
        mu_a       = np.mean(A, axis=0)

        # JSD aligned mean
        min_jsd = min(len(j) for j in seed_data["jsd"])
        mu_jsd  = np.mean([j[:min_jsd] for j in seed_data["jsd"]], axis=0)

        # ── Four-subplot validation curves (your std shadow architecture) ────────────────────
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        fig.suptitle(f"Validation: {label} ({num_seeds} seeds ± 1 STD)",
                     fontsize=13, fontweight="bold")

        t_full = np.arange(rounds)
        t_jsd  = np.arange(min_jsd)

        # ① Reward + std shadow
        axes[0].plot(t_full, mu_r, color="#3498db", lw=1.5, label="Mean Reward")
        axes[0].fill_between(t_full, mu_r - sd_r, mu_r + sd_r,
                             color="#3498db", alpha=0.2, label="±1 STD")
        axes[0].set_ylabel("Joint Reward")
        # Mark shuffle (use the first seed as representative)
        sr = seed_data["shuffle_rounds"][0]
        if sr:
            axes[0].scatter(sr, mu_r[sr], marker="*", color="#e67e22",
                           s=60, zorder=5, label="Shuffle (seed 0)")
        axes[0].legend(loc="upper left", fontsize=8)

        # ② Observer Accuracy
        axes[1].plot(t_full, mu_a, color="#9b59b6", lw=1.5)
        axes[1].axhline(ACC_TRANSPARENT_THRESH, ls="--", color="gray",
                       lw=0.8, label=f"Transparent Threshold ({ACC_TRANSPARENT_THRESH})")
        axes[1].axhline(1.0, ls=":", color="green", lw=0.8, label="Perfect Acc")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Observer Accuracy")
        axes[1].legend(loc="lower right", fontsize=8)

        # ③ JSD Rep Shift (fixes blind spot 1)
        axes[2].plot(t_jsd, mu_jsd, color="#1abc9c", lw=1.2)
        axes[2].axhline(JSD_STABLE_THRESH, ls="--", color="gray",
                       lw=0.8, label=f"Stable Threshold ({JSD_STABLE_THRESH})")
        axes[2].set_ylabel("JSD Rep Shift")
        axes[2].set_ylim(0, 0.75)
        axes[2].legend(fontsize=8)

        # ④ Collusion Score (log scale)
        min_cs = min(len(c) for c in seed_data["cs"])
        mu_cs  = np.mean([c[:min_cs] for c in seed_data["cs"]], axis=0)
        axes[3].plot(np.arange(min_cs), mu_cs, color="#e74c3c", lw=1.0, alpha=0.8)
        axes[3].axhline(ts, ls="--", color="#8e44ad", lw=0.8, label=f"Shuffle threshold ({ts})")
        axes[3].axhline(tp, ls="--", color="#e67e22", lw=0.8, label=f"Penalty threshold ({tp})")
        axes[3].set_yscale("log")
        axes[3].set_ylabel("Collusion Score (log)")
        axes[3].set_xlabel("Round")
        axes[3].legend(fontsize=8)

        plt.tight_layout()
        fig_path = f"figures/validate_{label}.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  -> Chart: {fig_path}")

        all_results[label] = {
            "final_mean_reward": float(np.mean(R[:, -100:])),
            "final_std_reward":  float(np.std( R[:, -100:])),
            "final_mean_acc":    float(np.mean(A[:, -WINDOW_SIZE:])),
            "seeds": {
                f"seed_{i}": {
                    "joint_rewards": seed_data["rewards"][i].tolist(),
                    "observer_accs": seed_data["accs"][i].tolist()
                } for i in range(num_seeds)
            }
        }

    # ── Welch t-test: compare two extreme configurations ─────────────────────────────
    keys = list(all_results.keys())
    if len(keys) >= 2:
        k1, k2 = keys[0], keys[-1]
        r1, r2 = all_results[k1], all_results[k2]
        t_stat, p_val = stats.ttest_ind_from_stats(
            mean1=r1["final_mean_reward"], std1=r1["final_std_reward"], nobs1=num_seeds,
            mean2=r2["final_mean_reward"], std2=r2["final_std_reward"], nobs2=num_seeds,
            equal_var=False
        )
        sig = "✅ p < 0.05" if p_val < 0.05 else "❌ n.s."
        print(f"\n[Welch t-test] {k1} vs {k2}")
        print(f"  t={t_stat:.3f}, p={p_val:.4f}  {sig}")
        all_results["stat_test"] = {
            "comparison": f"{k1} vs {k2}",
            "t": float(t_stat), "p": float(p_val), "significant": bool(p_val < 0.05)
        }

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Three Regime Comparison Plot (Core figure of the paper)
# ═══════════════════════════════════════════════════════════════════════

def plot_regime_comparison(results: Dict,
                            output_fig: str = "figures/regime_comparison.png"):
    """Reward + Acc comparison of three typical configurations, including std shadow and statistical annotations"""
    configs = [
        ("tp=3.0_ts=5.0",   PHASE_COLORS["collapse"],         "Collapse (3/5)"),
        ("tp=9.0_ts=14.0",  PHASE_COLORS["boundary"],         "Boundary (9/14)"),
        ("tp=12.0_ts=18.0", PHASE_COLORS["transparent"],      "Transparent (12/18)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Three Regime Comparison", fontsize=13, fontweight="bold")

    for key, color, label in configs:
        if key not in results:
            continue
        r   = results[key]
        mu  = np.array(r.get("reward_mean_curve", []))
        sd  = np.array(r.get("reward_std_curve",  []))
        acc = np.array(r.get("acc_mean_curve",    []))
        if len(mu) == 0:
            continue
        x = np.arange(len(mu))
        axes[0].plot(x, mu, color=color, lw=1.5, label=label)
        axes[0].fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15)
        axes[1].plot(x, acc, color=color, lw=1.5, label=label)

    axes[0].set_title("Joint Reward (Mean ± STD)")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].set_title("Observer Accuracy")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, ls="--", color="gray", lw=0.8)
    axes[1].legend()

    if "stat_test" in results:
        st = results["stat_test"]
        sig_txt = "✓ p<0.05" if st["significant"] else "✗ n.s."
        axes[0].annotate(
            f"{st['comparison']}\n{sig_txt} (t={st['t']:.2f}, p={st['p']:.4f})",
            xy=(0.02, 0.05), xycoords="axes fraction", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(output_fig, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[Plot] Regime comparison → {output_fig}")


# ═══════════════════════════════════════════════════════════════════════
# Main Program Entry
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIM Collusion Detection Framework")
    parser.add_argument("--mode", choices=["sweep", "validate", "full"], default="full",
                        help="sweep=Scan phase diagram | validate=Deep validation | full=Run both")
    parser.add_argument("--seeds",  type=int, default=5)
    parser.add_argument("--rounds", type=int, default=ROUNDS)

    parser.add_argument('--tp', '--threshold_penalty', type=float, default=7.5)
    parser.add_argument('--ts', '--threshold_shuffle', type=float, default=11.0)
    parser.add_argument('--alpha_ema', type=float, default=0.2)
    # ── Anti-Collusion Penalty Parameters (passed through to vqvae_agents_AIM.py) ──
    parser.add_argument('--penalty_base',          type=float, default=4.0)
    parser.add_argument('--penalty_factor',        type=float, default=1.0)
    parser.add_argument('--max_penalty',           type=float, default=15.0)
    parser.add_argument('--freeze_duration',       type=int,   default=15)
    parser.add_argument('--freeze_cooldown',       type=int,   default=20)
    parser.add_argument('--survival_reward',       type=float, default=0.1)
    parser.add_argument('--gradient_penalty_coeff',type=float, default=2.5)
    # ── Observer Strategy (Experimental V2.0) ──────────────────────────────
    parser.add_argument('--observer_strategy', type=str, default='fixed',
                        choices=['fixed', 'randomized_pool'],
                        help=(
                            'Observer evaluator strategy. '
                            '"fixed" = single BaselineObserver (default, matches all V1.5 experiments). '
                            '"randomized_pool" = experimental anti-gaming pool (V2.0, not yet validated).'
                        ))
    parser.add_argument('--observer_pool_size',          type=int,   default=8)
    parser.add_argument('--observer_subset_size',        type=int,   default=3)
    parser.add_argument('--observer_refresh_interval',   type=int,   default=100)
    parser.add_argument('--observer_lr',                 type=float, default=1e-3)
    parser.add_argument('--output_file', type=str, default='validation_results.json',
                        help='Output JSON filename for validation results (enables parallel runs)')

    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    if args.tp and args.ts:
        SPECIFIC_THRESHOLDS = [(args.tp, args.ts)]
    else:
        SPECIFIC_THRESHOLDS = [(3.0, 5.0), (9.0, 14.0), (12.0, 18.0)]

    if args.mode in ("sweep", "full"):
        print("=" * 60)
        print("Step 5: Threshold Sweep -> Phase Diagram")
        print("=" * 60)
        sweep_result = perform_threshold_sweep(
            penalty_range=(3.0, 18.0),
            shuffle_range=(5.0, 25.0),
            num_seeds=args.seeds,
            grid_points=10,
            observer_strategy=args.observer_strategy,
            observer_pool_size=args.observer_pool_size,
            observer_subset_size=args.observer_subset_size,
            observer_refresh_interval=args.observer_refresh_interval,
            observer_lr=args.observer_lr,
        )

    if args.mode in ("validate", "full"):
        print("\n" + "=" * 60)
        print("Step 6: Validate Specific Thresholds")
        print("=" * 60)
        val_results = validate_thresholds(
            specific_thresholds=SPECIFIC_THRESHOLDS,
            num_seeds=args.seeds,
            rounds=args.rounds,
            observer_strategy=args.observer_strategy,
            observer_pool_size=args.observer_pool_size,
            observer_subset_size=args.observer_subset_size,
            observer_refresh_interval=args.observer_refresh_interval,
            observer_lr=args.observer_lr,
        )
        with open(args.output_file, "w") as f:
            json.dump(val_results, f, indent=2)
        print(f"✅ Results saved to {args.output_file}")

    print("\n✅ Done! All charts are in the figures/ folder")
