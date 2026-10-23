import os
import yaml
from ultralytics import YOLO

def load_yaml_names(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return names

models = ["yolov8n_canonical_e100", "yolo12n_canonical_e100", "yolo26n_canonical_e100"]
base_dir = "runs/detect/runs/detect/timpla_full"
data_yaml = "timpla_combined/data.yaml"

print(f"{'Model':<25} | {'mAP50':<10} | {'mAP50-95':<10}")
print("-" * 50)

for m in models:
    weights_path = os.path.join(base_dir, m, "weights/best.pt")
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        # Run validation on the canonical dataset
        results = model.val(data=data_yaml, imgsz=640, device="mps", plots=False, verbose=False)
        print(f"{m:<25} | {results.box.map50:<10.4f} | {results.box.map:<10.4f}")
    else:
        print(f"{m:<25} | Weights not found")
