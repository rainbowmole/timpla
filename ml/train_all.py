import os
import sys
from ultralytics import YOLO

EPOCHS = int(sys.argv[1])
MODELS = sys.argv[2:] or ["yolov8n.pt", "yolo26n.pt", "yolo12n.pt"]
DEVICE = os.getenv("TIMPLA_DEVICE", "mps")

results = {}
for weights in MODELS:
    name = f"{weights.replace('.pt', '')}_e{EPOCHS}"
    try:
        model = YOLO(weights)
        model.train(
            data="timpla_combined/data.yaml", epochs=EPOCHS, imgsz=640,
            batch=16, device=DEVICE, project="timpla_runs", name=name,
            patience=15, seed=0,
        )
        m = model.val(device=DEVICE)
        results[name] = (m.box.map50, m.box.map)
    except Exception as e:
        results[name] = f"FAILED: {e}"

print("\n=== SUMMARY ===")
for name, r in results.items():
    print(f"{name}: mAP50={r[0]:.3f}  mAP50-95={r[1]:.3f}" if isinstance(r, tuple) else f"{name}: {r}")