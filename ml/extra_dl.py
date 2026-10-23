"""
Download the two extra Roboflow sources into datasets/.
Run from the project root (venv active):  python ml/download_extra.py

Before running, open each dataset page on Roboflow Universe and check:
  * the version number (Versions tab). Prefer a version WITHOUT augmentation,
    because augmented versions add near-duplicate images.
  * the license (also saved in README.dataset.txt inside each download).
    Record the attribution for your thesis.
"""
import os
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# (workspace, project, version, destination folder)
SOURCES = [
    ("yolo-1eope", "pantry-wlooe", 1, "datasets/pantry"),
    ("thanh-huy-phan", "food-ingredient-recognition", 1, "datasets/thp_ingredients"),
]

if __name__ == "__main__":
    for workspace, project, version, dest in SOURCES:
        if os.path.isdir(dest):
            print(f"[skip] {dest} already exists (delete it to re-download)")
        else:
            print(f"Downloading {workspace}/{project} v{version} -> {dest}")
            rf.workspace(workspace).project(project).version(version).download(
                "yolov8", location=dest
            )
        with open(os.path.join(dest, "data.yaml")) as f:
            names = yaml.safe_load(f)["names"]
        print(f"  {len(names)} classes: {names}\n")