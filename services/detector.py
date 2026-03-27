"""
Tiện ích để tải và chạy mô hình YOLO dự đoán các vùng `bbox` và `id`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Represents a single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Sequence[float]  # [x1, y1, x2, y2]


class BBoxIDDetector:
    """Thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self._model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold

    @property
    def class_names(self) -> dict[int, str]:
        return getattr(self._model.model, "names", self._model.names)

    def predict(
        self, image: Image.Image, conf: float | None = None, iou: float = 0.5, max_det: int = 100
    ) -> List[Detection]:
        """Run inference on a PIL image and return detections."""

        if conf is None:
            conf = self.conf_threshold

        # YOLO expects numpy arrays
        np_image = np.array(image.convert("RGB"))
        results = self._model.predict(np_image, conf=conf, iou=iou, max_det=max_det, verbose=False)

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = self.class_names.get(cls_id, str(cls_id))
            confidence = float(box.conf.item()) if box.conf is not None else 0.0
            xyxy = box.xyxy.tolist()
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=xyxy[0] if xyxy and isinstance(xyxy[0], (list, tuple)) else xyxy,
                )
            )

        return detections
