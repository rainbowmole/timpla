from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "runs/detect/timpla_runs/yolo12n_e50-3/weights/best.pt"


class IngredientDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._model: Any = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO

            if not self.model_path.exists():
                raise FileNotFoundError(f"YOLO weights not found: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image_bytes: bytes) -> list[dict]:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = self._load_model().predict(source=image, verbose=False)[0]
        names = result.names
        counts = Counter(int(cls) for cls in result.boxes.cls.tolist())
        detections = []
        for box, confidence, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            class_id = int(cls)
            label = names[class_id]
            detections.append(
                {
                    "ingredient_id": label,
                    "label": label,
                    "confidence": round(float(confidence), 4),
                    "bbox": [round(float(value), 2) for value in box],
                    "count": counts[class_id],
                }
            )
        return detections
