# Timpla: Filipino Ingredient Recognition

A professional machine learning pipeline for the detection and classification of Filipino ingredients using YOLOv12n.

## 🚀 Model Implementation Guide

To implement the Timpla detection model in your own system, follow these steps:

### 1. Requirements
Ensure you have the following environment:
- **Python**: 3.10 - 3.12
- **Dependencies**: Install the pinned requirements to ensure version compatibility.
  ```bash
  pip install -r requirements.txt
  ```

### 2. Model Weights
The champion model is trained on the 68-class expanded dataset. 
- **Architecture**: YOLOv12n
- **Weights**: Download the latest `best.pt` from the [GitHub Releases](https://github.com/rainbowmole/timpla/releases) page.
- **Input Size**: `640x640`

### 3. Inference Implementation
Using the `ultralytics` library, you can implement detection as follows:

```python
from ultralytics import YOLO

# Load the champion model
model = YOLO('path/to/best.pt')

# Run inference
# Recommended settings for Filipino ingredients:
# - conf=0.35 (Balance between precision and recall)
# - iou=0.5 (Prevents over-merging of overlapping ingredients)
results = model.predict(source='image.jpg', conf=0.35, iou=0.5, imgsz=640)

# Access detections
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"Detected {label} with {confidence:.2f} confidence")
```

### 4. Taxonomy & IDs
The model uses a **68-class canonical list**. To ensure your system maps detections correctly, always reference the `canonical_list.txt` or `data.yaml` in this repository. **Never map detections by integer ID**, as IDs may shift between dataset versions; always map by the class name string.

---

## 🛠️ Development & Training

### Machine Learning Tooling
The [`ml/`](ml/) directory contains the full pipeline for dataset merging, auditing, and evaluation.

### Training with Docker
Build the training image:
```bash
docker build -f ml/Dockerfile -t timpla-ml .
docker run --rm -v "$PWD:/workspace" timpla-ml 10
```

### Local Testing
To test the model visually on your local machine, use the provided detector app:
```bash
# Activate your environment
source .detector_env/bin/activate
streamlit run ml/timpla_detector_app.py
```
