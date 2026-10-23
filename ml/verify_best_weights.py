import pandas as pd
import os
from ultralytics import YOLO
import yaml

def get_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # final epoch
    final = df.iloc[-1]
    # best epoch - the one with the highest mAP50 (col index 7 in results.csv)
    # In Ultralytics results.csv: epoch, time, train/box, train/cls, train/dfl, metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95...
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best = df.loc[best_idx]
    return int(best.name + 1), best['metrics/mAP50(B)'], best['metrics/mAP50-95(B)'], \
           int(final.name + 1), final['metrics/mAP50(B)'], final['metrics/mAP50-95(B)']

models = ["yolov8n_canonical_e100", "yolo12n_canonical_e100", "yolo26n_canonical_e100"]
base_dir = "runs/detect/runs/detect/timpla_full"
data_yaml = "timpla_combined/data.yaml"

print(f"{'Model':<25} | {'Best Ep':<8} | {'B-mAP50':<10} | {'B-mAP95':<10} | {'F-mAP50':<10} | {'F-mAP95':<10}")
print("-" * 80)

for m in models:
    csv = os.path.join(base_dir, m, "results.csv")
    if os.path.exists(csv):
        b_ep, b_50, b_95, f_ep, f_50, f_95 = get_metrics(csv)
        print(f"{m:<25} | {b_ep:<8} | {b_50:<10.4f} | {b_95:<10.4f} | {f_50:<10.4f} | {f_95:<10.4f}")
    else:
        print(f"{m:<25} | NOT FOUND")

