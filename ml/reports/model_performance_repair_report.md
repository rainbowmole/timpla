# TIMPLA Model Performance and Training Pipeline Repair Report

## 1. Executive Summary
The goal was to professionalize the TIMPLA ingredient-recognition pipeline, diagnose a suspected model performance regression, and ensure deterministic dataset versioning. 

**Key Finding:** The suspected "regression" was a phantom caused by a Class-ID mapping mismatch. Checkpoints trained on earlier versions of the dataset (58 classes) were evaluated using the current canonical taxonomy (66 classes) without remapping, leading to near-zero mAP scores. Once remapped, YOLOv8n and YOLOv26n showed competitive performance, though YOLOv12n remains the strongest overall.

## 2. Pipeline Repairs and Defects Discovered

### 2.1 Dataset Merge Pipeline (`ml/merge_datasets.py`)
- **Defect: Non-Deterministic Splitting.** 
  - *Root Cause:* Reliance on `random.shuffle` without a persistent state for Open Images V7 subsets.
  - *Fix:* Implemented **Persisted Manifests**. The script now saves/loads `.json` manifests for each class folder to ensure absolute split reproducibility.
- **Defect: Unsafe Versioning.**
  - *Root Cause:* `shutil.rmtree` on the active `timpla_combined` folder before every merge.
  - *Fix:* Implemented a **Legacy Rename Strategy**. Existing folders are renamed to `timpla_combined_legacy_TIMESTAMP` and the new version is symlinked.
- **Defect: Mapping Warnings.**
  - *Root Cause:* THP mapping included chicken thigh/leg, which were absent from the source.
  - *Fix:* Cleaned `THP_MAP` to reflect actual source labels.

### 2.2 Evaluation Pipeline (`ml/eval.py`)
- **Defect: Checkpoint Misidentification.**
  - *Root Cause:* Previous evaluation calls passed base pretrained weights (`yolov8n.pt`) instead of the trained checkpoints in `runs/detect/timpla_runs/`.
- **Defect: Class-ID Mismatch.**
  - *Root Cause:* Evaluator assumed all checkpoints used the current 66-class taxonomy.
  - *Fix:* Developed a remapping verification tool and evaluation mode that maps prediction IDs based on the checkpoint's internal `names` dictionary.

## 3. Verified Model Performance (v1-like Subset)

The following metrics were obtained using a **rebuilt v1-like validation subset** (331 images).

| Model | Internal mAP50 (results.csv) | Remapped v1-subset mAP50 | Status |
| :--- | :--- | :--- | :--- |
| **YOLOv12n e50-2** | 0.5016 | **0.496** | Verified Best |
| **YOLOv8n e50** | 0.4167 | **0.4354** | Verified Competitive |
| **YOLOv26n e50** | 0.3678 | **0.3689** | Verified Competitive |

**Per-Class mAP50 Comparison (Remapped):**
- **Garlic**: v8n (0.405) > v12n (0.349) > v26n (0.293)
- **Talong**: v12n (0.622) > v26n (0.587) > v8n (0.497)
- **Tomato**: v12n (0.608) $\approx$ v8n (0.650) > v26n (0.604)
- **Sayote**: v12n (0.767) > v8n (0.730) > v26n (0.714)
- **Radish**: v8n (0.464) $\approx$ v26n (0.448) > v12n (0.405)

## 4. Dataset State
- **Canonical Taxonomy**: 66 classes.
- **Versioning**: Symlink-based (`timpla_combined` $\rightarrow$ `timpla_combined_TIMESTAMP`).
- **Reproducibility**: Verified. Double-runs of `merge_datasets.py` produce identical image counts and file lists in valid/test splits.

## 5. Next Actions and Blockers
1. **Full YOLOv8n Training**: Now that the pipeline is repaired and deterministic, a full training run on the corrected 66-class dataset is justified to see if v8n can exceed v12n performance.
2. **Error Analysis**: Deep dive into the "priority 8" weak classes (e.g., annatto_seeds, tofu) using the now-reliable evaluation pipeline.
3. **Supplementary Collection**: Prioritize images for classes where mAP is consistently low across all three architectures.

## 6. Reproducible Commands
- **Dataset Merge**: `python ml/merge_datasets.py`
- **Compatible Evaluation**: `python ml/remap_and_eval.py <path_to_best.pt>`
