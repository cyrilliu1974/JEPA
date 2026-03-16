"""
aim_bridge/video_dataset.py
============================
影片資料載入模組

功能：
  1. KineticsMiniDataset  - 載入 Kinetics-mini 影片，按類別分組
  2. VideoDataLoader      - 建立 DataLoader 供量化器訓練使用
  3. build_condition_dict - 把影片分組成干預實驗所需的格式

對接關係：
  - 輸出格式直接對接 stage1_diagnosis.py 的 run_full_diagnosis()
  - 使用 V-JEPA 2 官方的 vjepa2_preprocessor 確保前處理一致

目錄結構假設：
  data/kinetics_mini/
  └── val/
      ├── archery/
      │   ├── video001.mp4
      │   └── ...
      ├── bowling/
      ├── flying_kite/
      ├── high_jump/
      └── marching/

使用方式：
  from video_dataset import build_condition_dict, build_train_loader

  condition_videos = build_condition_dict(
      video_root="./data/kinetics_mini/val",
      processor=processor,
      device="cuda",
  )
  train_loader = build_train_loader(
      video_root="./data/kinetics_mini/val",
      processor=processor,
  )
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# ── 支援的影片副檔名 ──────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


# ═══════════════════════════════════════════════════════════════════════
# 影片解碼工具
# 按優先順序嘗試不同的解碼後端
# ═══════════════════════════════════════════════════════════════════════

def _load_video_frames(
    video_path: str,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[torch.Tensor]:
    """
    載入影片並採樣指定幀數。
    按優先順序嘗試：torchcodec → decord → torchvision → opencv

    Returns:
        tensor [C, T, H, W]，float32，範圍 [0, 1]
        失敗時返回 None
    """

    # ── 嘗試 torchcodec（V-JEPA 2 官方推薦）──────────────────────────
    try:
        from torchcodec.decoders import VideoDecoder
        import numpy as np

        decoder = VideoDecoder(video_path)
        total = len(decoder)
        if total == 0:
            raise ValueError("Empty video")

        # 均勻採樣 num_frames 幀
        indices = torch.linspace(0, total - 1, num_frames).long().tolist()
        frames_data = decoder.get_frames_at(indices=indices)
        
        # 處理不同版本的 torchcodec (可能是 FrameBatch 物件或直接是 tensor)
        if hasattr(frames_data, 'data'):
            frames = frames_data.data
        else:
            frames = frames_data

        # 轉換格式
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        
        frames = frames.float() / 255.0

        # torchcodec 可能回傳 [T, H, W, C] 或 [T, C, H, W]
        if frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
        else:
            frames = frames.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(num_frames, *target_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return frames

    except ImportError:
        pass
    except Exception as e:
        log.debug(f"torchcodec failed for {video_path}: {e}")

    # ── 嘗試 decord ───────────────────────────────────────────────────
    try:
        import decord
        import numpy as np
        decord.bridge.set_bridge("torch")

        vr = decord.VideoReader(video_path, width=target_size[1], height=target_size[0])
        total = len(vr)
        if total == 0:
            raise ValueError("Empty video")

        indices = torch.linspace(0, total - 1, num_frames).long().tolist()
        frames = vr.get_batch(indices)   # [T, H, W, C]
        frames = frames.float() / 255.0
        frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
        return frames

    except ImportError:
        pass
    except Exception as e:
        log.debug(f"decord failed for {video_path}: {e}")

    # ── 嘗試 torchvision ─────────────────────────────────────────────
    try:
        import torchvision.io as tvio
        import numpy as np

        frames, _, _ = tvio.read_video(video_path, pts_unit="sec")
        # frames: [T, H, W, C]
        if frames.shape[0] == 0:
            raise ValueError("Empty video")

        total = frames.shape[0]
        indices = torch.linspace(0, total - 1, num_frames).long()
        frames = frames[indices]                   # [T, H, W, C]
        frames = frames.float() / 255.0
        frames = frames.permute(3, 0, 1, 2)       # [C, T, H, W]
        frames = F.interpolate(
            frames.unsqueeze(0),
            size=(num_frames, *target_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return frames

    except ImportError:
        pass
    except Exception as e:
        log.debug(f"torchvision failed for {video_path}: {e}")

    # ── 嘗試 OpenCV（最後手段）───────────────────────────────────────
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            raise ValueError("Empty video")

        indices = set(torch.linspace(0, total - 1, num_frames).long().tolist())
        frame_list = []
        frame_idx = 0

        while cap.isOpened() and len(frame_list) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
                frame_list.append(torch.from_numpy(frame))
            frame_idx += 1
        cap.release()

        if not frame_list:
            raise ValueError("No frames extracted")

        # 補齊不足的幀
        while len(frame_list) < num_frames:
            frame_list.append(frame_list[-1])

        frames = torch.stack(frame_list[:num_frames])  # [T, H, W, C]
        frames = frames.float() / 255.0
        frames = frames.permute(3, 0, 1, 2)            # [C, T, H, W]
        return frames

    except ImportError:
        pass
    except Exception as e:
        log.debug(f"opencv failed for {video_path}: {e}")

    log.error(f"All decoders failed for: {video_path}")
    log.error("Please install: pip install torchcodec  OR  pip install decord")
    return None


def _apply_vjepa2_preprocessor(
    frames: torch.Tensor,
    processor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    用 V-JEPA 2 官方 preprocessor 處理影片。

    如果 processor 是 None（mock 模式），直接返回原始 tensor。
    frames: [C, T, H, W]
    Returns: tensor，格式符合 encoder 要求
    """
    if processor is None:
        # Mock 模式：直接返回，假設格式已正確
        return frames.unsqueeze(0)  # [1, C, T, H, W]

    try:
        # V-JEPA 2 processor 期望 [T, H, W, C] 格式的 numpy 或 tensor
        # frames 目前是 [C, T, H, W]
        frames_thwc = frames.permute(1, 2, 3, 0)  # [T, H, W, C]
        frames_uint8 = (frames_thwc * 255).byte()

        # convert to numpy list to avoid "Unable to infer channel dimension format"
        video_np = list(frames_uint8.cpu().numpy())

        processed = processor(video_np, return_tensors="pt")
        # 取出 pixel_values
        if hasattr(processed, "pixel_values"):
            return processed.pixel_values.to(device)
        else:
            return processed["pixel_values"].to(device)



    except Exception as e:
      
        # 手動套用 ImageNet 標準化（V-JEPA 2 預訓練使用的標準）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames - mean) / std   # frames 已經是 [0,1] 範圍
        return frames.to(device)




# ═══════════════════════════════════════════════════════════════════════
# Dataset 類別
# ═══════════════════════════════════════════════════════════════════════

class KineticsMiniDataset(Dataset):
    """
    載入 Kinetics-mini 影片資料集。

    目錄結構：
        video_root/
        ├── archery/
        ├── bowling/
        ├── flying_kite/
        ├── high_jump/
        └── marching/

    每個子目錄對應一個動作類別（干預條件）。
    """

    # Kinetics-mini 的五個類別與對應的語義描述
    CATEGORY_SEMANTICS = {
        "archery":     "精確上肢控制、靜止釋放",
        "bowling":     "投擲物體、碰撞動力學",
        "flying_kite": "外力場景、流體力學",
        "high_jump":   "跳躍、重力、姿態變化",
        "marching":    "週期性步態、群體運動",
    }

    def __init__(
        self,
        video_root: str,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        processor=None,
        device: str = "cpu",
        max_per_class: int = 50,        # 每個類別最多載入幾個影片
        min_per_class: int = 5,         # 低於此數量的類別發出警告
    ):
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.target_size = target_size
        self.processor = processor
        self.device = device

        # 掃描所有影片
        self.samples: List[Tuple[str, int, str]] = []  # (path, class_idx, class_name)
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._scan_directory(max_per_class, min_per_class)
        log.info(f"KineticsMiniDataset: {len(self.samples)} videos, "
                 f"{len(self.class_names)} classes")

    def _scan_directory(self, max_per_class, min_per_class):
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        # 找所有子目錄（每個子目錄是一個類別）
        class_dirs = sorted([
            d for d in self.video_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.video_root}")

        for class_dir in class_dirs:
            class_name = class_dir.name
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = len(self.class_names) - 1

            # 找所有影片檔案
            video_files = sorted([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in VIDEO_EXTENSIONS
            ])[:max_per_class]

            if len(video_files) < min_per_class:
                log.warning(
                    f"Class '{class_name}' has only {len(video_files)} videos "
                    f"(minimum recommended: {min_per_class})"
                )

            for vf in video_files:
                self.samples.append((
                    str(vf),
                    self.class_to_idx[class_name],
                    class_name,
                ))

        log.info(f"Classes found: {self.class_names}")
        for cn in self.class_names:
            count = sum(1 for _, _, name in self.samples if name == cn)
            semantic = self.CATEGORY_SEMANTICS.get(cn, "（未知語義）")
            log.info(f"  {cn}: {count} videos  [{semantic}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, class_idx, class_name = self.samples[idx]

        frames = _load_video_frames(
            video_path, self.num_frames, self.target_size
        )

        if frames is None:
            # 載入失敗時返回零張量，避免 DataLoader 崩潰
            log.warning(f"Failed to load: {video_path}, using zeros")
            frames = torch.zeros(3, self.num_frames, *self.target_size)

        processed = _apply_vjepa2_preprocessor(
            frames, self.processor, self.device
        )

        # 確保輸出格式是 [C, T, H, W]（去掉 batch 維度）
        if processed.dim() == 5:
            processed = processed.squeeze(0)

        return processed, class_idx

    def get_class_videos(
        self,
        class_name: str,
        as_tensors: bool = True,
    ) -> List:
        """
        取得特定類別的所有影片。
        返回：List[Tensor]，每個 Tensor 是 [C, T, H, W]
        """
        indices = [
            i for i, (_, _, name) in enumerate(self.samples)
            if name == class_name
        ]
        if not indices:
            raise ValueError(f"Class '{class_name}' not found. "
                           f"Available: {self.class_names}")

        if not as_tensors:
            return [self.samples[i][0] for i in indices]  # 返回路徑

        tensors = []
        for i in indices:
            tensor, _ = self[i]
            tensors.append(tensor.unsqueeze(0))  # [1, C, T, H, W]
        return tensors


# ═══════════════════════════════════════════════════════════════════════
# 工具函數：建立干預實驗所需的資料格式
# ═══════════════════════════════════════════════════════════════════════

def build_condition_dict(
    video_root: str,
    processor=None,
    device: str = "cpu",
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    max_per_class: int = 50,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    建立 stage1_diagnosis.py 的 run_full_diagnosis() 所需的資料格式。

    返回：
    {
        "variable_A": {"archery": [tensor, ...], "bowling": [tensor, ...]},
        "variable_B": {"flying_kite": [...], "high_jump": [...]},
        "variable_C": {"marching": [...]},
        "stability":  [tensor, tensor, ...]   ← H1 測試用
    }

    Kinetics-mini 五個類別的分組策略：
    - variable_A（動作精確度）：archery vs bowling
    - variable_B（空間動態）：flying_kite vs high_jump
    - variable_C（運動模式）：marching（單獨作為週期性運動基準）
    - stability：從所有類別各取 1-2 個影片
    """
    dataset = KineticsMiniDataset(
        video_root=video_root,
        num_frames=num_frames,
        target_size=target_size,
        processor=processor,
        device=device,
        max_per_class=max_per_class,
    )

    available = dataset.class_names
    log.info(f"\nBuilding condition dict from: {available}")

    def safe_get(class_name, max_n=None):
        """安全取得類別影片，類別不存在時返回空列表"""
        if class_name not in available:
            log.warning(f"Class '{class_name}' not in dataset, skipping")
            return []
        videos = dataset.get_class_videos(class_name)
        if max_n:
            videos = videos[:max_n]
        return videos

    # ── Variable A：動作精確度對比 ─────────────────────────────────
    # archery（靜止、精確、目標導向）vs bowling（動態、衝擊、物理碰撞）
    variable_A = {}
    for cls in ["archery", "bowling"]:
        vids = safe_get(cls)
        if vids:
            variable_A[cls] = vids

    # ── Variable B：空間動態對比 ───────────────────────────────────
    # flying_kite（外力主導、非接觸）vs high_jump（肌肉主導、重力對抗）
    variable_B = {}
    for cls in ["flying_kite", "high_jump"]:
        vids = safe_get(cls)
        if vids:
            variable_B[cls] = vids

    # ── Variable C：運動模式 ───────────────────────────────────────
    # marching（週期性）vs archery（非週期性）
    # 如果 archery 已在 A，這裡重複使用沒問題（不同實驗獨立）
    variable_C = {}
    for cls in ["marching", "archery"]:
        vids = safe_get(cls)
        if vids:
            variable_C[cls] = vids

    # ── Stability 測試影片（H1 用）────────────────────────────────
    stability = []
    for cls in available[:3]:  # 從前三個類別各取 2 個
        vids = safe_get(cls, max_n=2)
        stability.extend(vids)

    # 統計
    log.info(f"\nCondition dict summary:")
    log.info(f"  variable_A: {list(variable_A.keys())} "
             f"({sum(len(v) for v in variable_A.values())} videos)")
    log.info(f"  variable_B: {list(variable_B.keys())} "
             f"({sum(len(v) for v in variable_B.values())} videos)")
    log.info(f"  variable_C: {list(variable_C.keys())} "
             f"({sum(len(v) for v in variable_C.values())} videos)")
    log.info(f"  stability:  {len(stability)} videos")

    return {
        "variable_A": variable_A,
        "variable_B": variable_B,
        "variable_C": variable_C,
        "stability":  stability,
    }


def build_train_loader(
    video_root: str,
    processor=None,
    device: str = "cpu",
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
    max_per_class: int = 50,
    num_workers: int = 0,       # 0 = 主進程，避免多進程問題
) -> DataLoader:
    """
    建立量化器訓練用的 DataLoader。

    注意：num_workers=0 是預設值，避免 macOS/Windows 的多進程問題。
    如果在 Linux 上跑，可以設為 2-4 加速資料載入。
    """
    dataset = KineticsMiniDataset(
        video_root=video_root,
        num_frames=num_frames,
        target_size=target_size,
        processor=processor,
        device="cpu",            # DataLoader 裡不在 GPU 上做，避免記憶體問題
        max_per_class=max_per_class,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    log.info(f"Train loader: {len(dataset)} videos, "
             f"batch_size={batch_size}, "
             f"{len(loader)} batches per epoch")
    return loader


# ═══════════════════════════════════════════════════════════════════════
# 影片下載工具
# ═══════════════════════════════════════════════════════════════════════

# 支援的資料集來源
DATASET_SOURCES = {
    "kinetics-mini": {
        "repo_id":      "nateraw/kinetics-mini",
        "repo_type":    "dataset",
        "pattern":      "val/*/*.mp4",
        "local_subdir": "kinetics_mini",
        "description":  "Kinetics-mini (~50 videos/class, ~5 classes)",
    },
    "kinetics": {
        "repo_id":      "nateraw/kinetics",
        "repo_type":    "dataset",
        "pattern":      "val/*/*.mp4",
        "local_subdir": "kinetics",
        "description":  "Kinetics-400 full (~400 videos/class, 400 classes)",
    },
}


def download_dataset(
    dataset_name: str = "kinetics-mini",
    local_dir: str = "./data",
    classes: Optional[List[str]] = None,
) -> str:
    """
    從 HuggingFace 下載影片資料集。

    參數：
        dataset_name : "kinetics-mini"（預設）或 "kinetics"
        local_dir    : 本地儲存根目錄（預設 ./data）
        classes      : 只下載指定類別，None 表示下載全部
                       例如：["archery", "bowling", "flying_kite"]
                       只在 dataset_name="kinetics" 時有效，
                       kinetics-mini 太小，直接下載全部。

    返回：
        下載後的 val 目錄路徑（可直接傳給 --video_root）

    使用範例：
        # 下載 kinetics-mini（預設，約 50 部/類）
        video_root = download_dataset()

        # 下載完整 kinetics 的特定類別（約 400 部/類）
        video_root = download_dataset(
            dataset_name="kinetics",
            classes=["archery", "bowling", "flying_kite", "high_jump", "marching"]
        )
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    if dataset_name not in DATASET_SOURCES:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available: {list(DATASET_SOURCES.keys())}"
        )

    source = DATASET_SOURCES[dataset_name]
    dest   = Path(local_dir) / source["local_subdir"]
    val_dir = dest / "val"

    # 如果已存在且有影片，跳過下載
    if val_dir.exists():
        existing = list(val_dir.rglob("*.mp4"))
        if existing:
            log.info(
                f"Dataset already exists at {val_dir} "
                f"({len(existing)} videos). Skipping download."
            )
            return str(val_dir)

    # 決定下載的 patterns
    if dataset_name == "kinetics-mini" or classes is None:
        # kinetics-mini 直接下載全部
        patterns = source["pattern"]
    else:
        # kinetics 只下載指定類別
        patterns = [f"val/{c}/*.mp4" for c in classes]
        log.info(f"Downloading classes: {classes}")

    log.info(f"Downloading {source['description']}...")
    log.info(f"  Source : {source['repo_id']}")
    log.info(f"  Dest   : {dest}")
    log.info(f"  Note   : This may take several minutes.")

    snapshot_download(
        repo_id=source["repo_id"],
        repo_type=source["repo_type"],
        local_dir=str(dest),
        allow_patterns=patterns,
    )

    n_downloaded = len(list(val_dir.rglob("*.mp4")))
    log.info(f"✅ Download complete: {n_downloaded} videos → {val_dir}")
    return str(val_dir)


# ═══════════════════════════════════════════════════════════════════════
# 環境檢查工具
# ═══════════════════════════════════════════════════════════════════════

def check_environment():
    """執行前先確認環境是否正確"""
    issues = []
    suggestions = []

    # 檢查影片解碼器
    decoders_available = []
    try:
        import torchcodec
        decoders_available.append("torchcodec ✅")
    except ImportError:
        pass

    try:
        import decord
        decoders_available.append("decord ✅")
    except ImportError:
        pass

    try:
        import torchvision
        decoders_available.append("torchvision ✅")
    except ImportError:
        pass

    try:
        import cv2
        decoders_available.append("opencv ✅")
    except ImportError:
        pass

    if not decoders_available:
        issues.append("No video decoder found")
        suggestions.append("pip install torchcodec  # recommended")
        suggestions.append("  OR: pip install decord")
        suggestions.append("  OR: pip install torchvision")
        suggestions.append("  OR: pip install opencv-python")
    else:
        log.info(f"Video decoders: {', '.join(decoders_available)}")

    # 檢查 transformers（V-JEPA 2 preprocessor 用）
    try:
        import transformers
        log.info(f"transformers ✅ ({transformers.__version__})")
    except ImportError:
        suggestions.append("pip install transformers  # for vjepa2_preprocessor")

    # 檢查 scipy（MI 和 JSD 計算用）
    try:
        import scipy
        log.info(f"scipy ✅ ({scipy.__version__})")
    except ImportError:
        issues.append("scipy not found")
        suggestions.append("pip install scipy")

    if issues:
        log.error("Environment issues found:")
        for issue in issues:
            log.error(f"  ❌ {issue}")
        log.info("Install suggestions:")
        for s in suggestions:
            log.info(f"  {s}")
        return False

    log.info("✅ Environment check passed")
    return True


# ═══════════════════════════════════════════════════════════════════════
# CLI 測試入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                       format="[%(asctime)s] %(message)s",
                       datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str,
                       default="./data/kinetics_mini/val")
    parser.add_argument("--check_env", action="store_true")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset from HuggingFace before scanning")
    parser.add_argument("--dataset", type=str, default="kinetics-mini",
                       choices=["kinetics-mini", "kinetics"],
                       help="Dataset to download (default: kinetics-mini)")
    parser.add_argument("--classes", type=str, nargs="+",
                       default=["archery", "bowling", "flying_kite",
                                "high_jump", "marching"],
                       help="Classes to download (only used when --dataset kinetics)")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Local directory to store downloaded data")
    args = parser.parse_args()

    if args.check_env:
        check_environment()
        sys.exit(0)

    # 下載資料集
    if args.download:
        video_root = download_dataset(
            dataset_name=args.dataset,
            local_dir=args.data_dir,
            classes=args.classes if args.dataset == "kinetics" else None,
        )
        args.video_root = video_root
        log.info(f"Using video_root: {args.video_root}")

    # 掃描資料集並顯示統計
    if not Path(args.video_root).exists():
        log.error(f"Video root not found: {args.video_root}")
        log.info("Download with:")
        log.info("  python video_dataset.py --download")
        log.info("  python video_dataset.py --download --dataset kinetics "
                 "--classes archery bowling flying_kite high_jump marching")
        sys.exit(1)

    check_environment()

    data = build_condition_dict(
        video_root=args.video_root,
        processor=None,
    )

    log.info("\n✅ Dataset ready for stage1_diagnosis.py")
    log.info("Next step:")
    log.info("  python stage1_diagnosis.py \\")
    log.info(f"      --video_root {args.video_root} \\")
    log.info("      --run_with_kinetics")
