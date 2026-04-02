"""EfficientNet 推理服务封装。"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .efficientnet import create_efficientnet
from .transforms import build_transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_CANDIDATES = [
    ROOT_DIR / "cloud_sync" / "efficientnet_b2" / "best_model.pt",
    ROOT_DIR / "outputs" / "smoke_efficientnet_b0" / "best_model.pt",
]


def choose_inference_device() -> torch.device:
    """自动选择推理设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(custom_path: str | Path | None = None) -> Path:
    """解析模型检查点路径。"""
    if custom_path is not None:
        checkpoint_path = Path(custom_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到模型文件：{checkpoint_path}")
        return checkpoint_path

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到可用的 EfficientNet 模型文件，请通过环境变量 MODEL_CHECKPOINT 指定路径。")


@dataclass
class InferenceResult:
    """单次推理结果。"""

    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    elapsed_ms: float
    image_size: tuple[int, int]


class EfficientNetInferenceService:
    """负责加载本地模型并执行推理。"""

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.device = choose_inference_device()
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.config = self.checkpoint["config"]
        self.idx_to_class = {int(k): v for k, v in self.checkpoint["idx_to_class"].items()}
        self.transform = build_transforms(image_size=self.config["image_size"], is_train=False)

        self.model = create_efficientnet(self.config["model_name"], num_classes=len(self.idx_to_class))
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_from_pil(self, image: Image.Image) -> InferenceResult:
        """对 PIL 图像执行推理。"""
        start_time = time.perf_counter()
        rgb_image = image.convert("RGB")
        original_size = rgb_image.size
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        probabilities = {
            self.idx_to_class[idx]: float(prob)
            for idx, prob in enumerate(probs.detach().cpu().tolist())
        }
        return InferenceResult(
            predicted_index=pred_idx,
            predicted_label=self.idx_to_class[pred_idx],
            confidence=probabilities[self.idx_to_class[pred_idx]],
            probabilities=probabilities,
            elapsed_ms=elapsed_ms,
            image_size=(original_size[0], original_size[1]),
        )

    def predict_from_bytes(self, image_bytes: bytes) -> InferenceResult:
        """对二进制图像执行推理。"""
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_from_pil(image)
