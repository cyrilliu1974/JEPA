"""
test_vjepa2_latent.py
=====================
診斷 V-JEPA 2 潛空間（z 向量）的數值統計。

目的：
  確認 V-JEPA 2 encoder 輸出的 z 向量數值範圍，
  用於調整 AIM Quantizer 的初始化參數。

使用方式：
  # Kaggle / Colab（有 GPU）
  !python test_vjepa2_latent.py --video_root ./data/kinetics_mini/val

  # 本機
  python test_vjepa2_latent.py --video_root C:/AI/JEPA/data/kinetics_mini/val

需要安裝：
  pip install av transformers torch torchvision
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ── 安裝 av（如果沒有）───────────────────────────────────────────────
try:
    import av
except ImportError:
    print("Installing PyAV...")
    os.system(f"{sys.executable} -m pip install av -q")
    import av

from transformers import AutoModel


def load_video_frames(video_path: str, num_frames: int = 16,
                      size: int = 224) -> torch.Tensor:
    """
    讀取影片並採樣 num_frames 幀。
    返回：[T, C, H, W] float32，已 resize 到 size x size，值域 [0, 1]
    """
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')           # [H, W, C]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_t = F.interpolate(img_t, size=(size, size),
                              mode='bilinear', align_corners=False)
        frames.append(img_t.squeeze(0))                  # [C, H, W]
        if len(frames) >= num_frames:
            break
    container.close()

    # 補幀（如果影片幀數不足）
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.stack(frames)   # [T, C, H, W]


def find_videos(video_root: str, n: int = 5):
    """從各類別各取一個影片，共取 n 個"""
    videos = []
    for cls in sorted(os.listdir(video_root)):
        cls_path = os.path.join(video_root, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in sorted(os.listdir(cls_path)):
            if f.endswith('.mp4'):
                videos.append(os.path.join(cls_path, f))
                break
        if len(videos) >= n:
            break
    return videos


def run_diagnosis(video_root: str, num_frames: int = 16, device: str = "cuda"):
    """
    完整診斷流程：
    1. 載入 V-JEPA 2 ViT-L
    2. 對多個影片取得 z 向量
    3. 輸出統計資訊
    4. 確認 temporal_pool 可行性
    """

    # ── GPU 確認 ─────────────────────────────────────────────────────
    print("=" * 50)
    print("環境確認")
    print("=" * 50)
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，切換到 CPU")
        device = "cpu"
    if device == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ℹ️  使用 CPU")

    # ── 載入 encoder ──────────────────────────────────────────────────
    print("\n載入 V-JEPA 2 ViT-L...")
    hf_repo = "facebook/vjepa2-vitl-fpc64-256"
    try:
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(hf_repo).to(device)
    except Exception as e:
        print(f"❌ HuggingFace 模型載入失敗 ({e})。")
        print("無法建構 V-JEPA 模型架構，因為您的 `transformers` 套件版本過舊。")
        print("請執行以下指令更新套件：")
        print("    pip install --upgrade transformers")
        print("\n更新後請重新執行此腳本。")
        return
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"✅ 載入完成")

    # ── 找影片 ───────────────────────────────────────────────────────
    print(f"\n掃描影片：{video_root}")
    videos = find_videos(video_root, n=5)
    if not videos:
        print(f"❌ 找不到 .mp4 影片：{video_root}")
        return
    print(f"✅ 找到 {len(videos)} 個影片")

    # ── 對每個影片取得 z 向量 ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Z 向量統計")
    print("=" * 50)

    all_z_stats = []

    for i, video_path in enumerate(videos):
        video_name = os.path.basename(video_path)
        print(f"\n[{i+1}/{len(videos)}] {video_name}")

        # 讀取影片
        frames = load_video_frames(video_path, num_frames, size=224)
        # [T, C, H, W] → [1, T, C, H, W]
        t = frames.unsqueeze(0).to(device)
        print(f"  Input [B, T, C, H, W]:  {t.shape}")
        
        # 準備 Local Checkpoint 需要的 [B, C, T, H, W] 格式
        t_local = t.permute(0, 2, 1, 3, 4).contiguous()

        # Encoder 推理
        with torch.no_grad():
            try:
                # 嘗試提供 keyword argument (HuggingFace AutoModel 方式)
                z = encoder(pixel_values_videos=t).last_hidden_state
            except TypeError:
                # TypeError 代表是不支援 keyword argument 的 Local Model
                out = encoder(t_local)
                if hasattr(out, 'last_hidden_state'):
                    z = out.last_hidden_state
                else:
                    z = out
        print(f"  Output: {z.shape}")

        # 統計
        stats = {
            "mean": z.mean().item(),
            "std":  z.std().item(),
            "min":  z.min().item(),
            "max":  z.max().item(),
            "norm": z.norm(dim=-1).mean().item(),
        }
        all_z_stats.append(stats)

        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
              f"norm={stats['norm']:.4f}")

    # ── 跨影片平均統計 ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("跨影片平均統計")
    print("=" * 50)
    for key in ["mean", "std", "min", "max", "norm"]:
        vals = [s[key] for s in all_z_stats]
        print(f"  {key:6s}: avg={np.mean(vals):.4f}  "
              f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    # ── temporal_pool 可行性確認 ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("Temporal Pool 可行性")
    print("=" * 50)
    B, N_tokens, D = z.shape
    print(f"  N_tokens = {N_tokens}")
    print(f"  num_frames = {num_frames}")
    divisible = N_tokens % num_frames == 0
    print(f"  N_tokens % num_frames == 0: {divisible}")

    if divisible:
        N_spatial = N_tokens // num_frames
        z_frames = z.reshape(B, num_frames, N_spatial, D).mean(dim=2)
        print(f"  ✅ temporal_pool 輸出: {z_frames.shape}")
        print(f"  frame-level std: {z_frames.std():.6f}")
    else:
        print(f"  ❌ 無法整除！需要調整 num_frames 參數")
        # 找可整除的幀數
        for nf in [8, 12, 16, 32, 64]:
            if N_tokens % nf == 0:
                print(f"     建議 num_frames={nf}")

    # ── Quantizer 參數建議 ────────────────────────────────────────────
    avg_std = np.mean([s["std"] for s in all_z_stats])
    avg_norm = np.mean([s["norm"] for s in all_z_stats])

    print("\n" + "=" * 50)
    print("AIM Quantizer 參數建議")
    print("=" * 50)
    print(f"  z 平均 std:  {avg_std:.4f}")
    print(f"  z 平均 norm: {avg_norm:.4f}")

    if avg_std < 0.01:
        print("  ⚠️  std 極小，可能導致 Loss=0（梯度消失）")
        print("  建議：降低 projection_dim 或提高 commitment_cost")
        print(f"  建議 commitment_cost: {min(2.0, 0.25 * (0.1 / avg_std)):.2f}")
    elif avg_std > 10.0:
        print("  ⚠️  std 過大，codebook 初始化需要調整")
        print("  建議：在 projection 之前加強 LayerNorm")
    else:
        print("  ✅ 數值範圍正常，現有參數應可正常訓練")

    print("\n✅ 診斷完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-JEPA 2 Latent Space Diagnosis")
    parser.add_argument("--video_root", type=str,
                        default="./data/kinetics_mini/val",
                        help="Kinetics-mini val 目錄路徑")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_diagnosis(args.video_root, args.num_frames, args.device)