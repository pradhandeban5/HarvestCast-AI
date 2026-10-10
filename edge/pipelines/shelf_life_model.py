"""Transfer-learning image model and checkpoint-backed shelf-life inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMAGE_SIZE = 160
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class MobileNetShelfLife(nn.Module):
    """Predict dataset shelf-life interval and Weibull survival parameters."""

    def __init__(self, class_count: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights)
        feature_count = self.encoder.classifier[-1].in_features
        self.encoder.classifier[-1] = nn.Identity()
        self.class_head = nn.Linear(feature_count, class_count)
        self.survival_head = nn.Linear(feature_count, 2)
        nn.init.normal_(self.class_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.class_head.bias)
        nn.init.normal_(self.survival_head.weight, mean=0.0, std=0.005)
        with torch.no_grad():
            self.survival_head.bias.copy_(torch.tensor([7.0, 0.5]))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return class logits plus positive Weibull scale and shape values."""
        embeddings = self.encoder(images)
        logits = self.class_head(embeddings)
        raw_parameters = self.survival_head(embeddings)
        scale_days = F.softplus(raw_parameters[:, 0]) + 0.05
        shape = F.softplus(raw_parameters[:, 1]) + 0.05
        return logits, scale_days, shape


class TrainedShelfLifePredictor:
    """Load a trained checkpoint and infer an interval-conditioned webcam RSL."""

    def __init__(self, checkpoint_path: Path, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.classes: list[dict[str, Any]] = checkpoint["classes"]
        self.validation_metrics: dict[str, Any] = checkpoint.get("validation_metrics", {})
        self.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        self.model = MobileNetShelfLife(len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    @classmethod
    def load_if_available(cls, checkpoint_path: Path) -> TrainedShelfLifePredictor | None:
        """Load the model only when a trained checkpoint exists."""
        if not checkpoint_path.is_file():
            return None
        return cls(checkpoint_path)

    @staticmethod
    def _canonical_product(produce: str) -> str:
        """Normalize dashboard plural names to the dataset's singular folders."""
        normalized = produce.strip().casefold().rstrip("s")
        return {"tomatoe": "tomato"}.get(normalized, normalized).title()

    def predict_bgr(self, frame_bgr: np.ndarray, produce: str) -> dict[str, Any]:
        """Predict a labeled range and Weibull RSL from a standard RGB webcam frame."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
            raise ValueError("Webcam input must be a non-empty BGR color frame")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, scale, shape = self.model(tensor)
            probabilities = torch.softmax(logits[0], dim=0)
        selected_product = self._canonical_product(produce)
        allowed = [
            index
            for index, item in enumerate(self.classes)
            if item["expired"] or item["product"] == selected_product
        ]
        if not allowed:
            allowed = list(range(len(self.classes)))
        conditional = probabilities[allowed]
        conditional = conditional / conditional.sum().clamp_min(1e-8)
        best_offset = int(torch.argmax(conditional).item())
        class_index = allowed[best_offset]
        prediction_class = self.classes[class_index]
        confidence = float(conditional[best_offset].item())
        weibull_scale = float(scale[0].item())
        weibull_shape = float(shape[0].item())
        expected_days = weibull_scale * math.gamma(1.0 + 1.0 / weibull_shape)

        if prediction_class["expired"]:
            expected_days = 0.0
            weibull_scale = 0.05
        else:
            lower = float(prediction_class["lower_days"])
            upper = float(prediction_class["upper_days"])
            expected_days = min(max(expected_days, lower), upper)
            weibull_scale = expected_days / math.gamma(1.0 + 1.0 / weibull_shape)

        return {
            "lambda_days": weibull_scale,
            "k_shape": weibull_shape,
            "expected_rsl_days": expected_days,
            "predicted_range": prediction_class["name"],
            "confidence": confidence,
            "model_status": "trained_interval_weibull",
        }