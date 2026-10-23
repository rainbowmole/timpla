# TIMPLA Project: Research & Engineering Handoff Report
**Date:** 2026-10-19
**Status:** Phase 1-7 Completed

## 1. Executive Summary
The project's data pipeline has been audited, repaired, and extended. We have moved from a non-deterministic merge process to a versioned, reproducible pipeline. A significant performance regression was diagnosed in newer model architectures compared to the YOLOv8 baseline on the original validation set. We have also implemented a new, provenance-aware recipe collection system.

## 2. Dataset Pipeline & Taxonomy
### Changes and Fixes
- **Deterministic Splitting**: Modified `ml/merge_datasets.py` to replace `random.shuffle` with sorted directory lists. Splits are now stable across runs.
- **Versioned Exports**: Implemented timestamped versioning (e.g., `timpla_combined_YYYYMMDD_HHMMSS`) with a symlink `timpla_combined` pointing to the latest version.
- **Mapping Audit**: Added a `dropped_labels.log` to track labels present in source data but missing from the canonical taxonomy.
- **Taxonomy Alignment**: Fixed mapping gaps in the THP dataset (specifically for chicken cuts).

### Dataset Statistics (Current Version)
- **Canonical Classes**: 66
- **Training Images**: 3,729
- **Validation Images**: 365
- **Test Images**: 229

### Validation Gaps (Priority for Collection)
The following classes have **zero** validation images and cannot be currently scored:
- `annatto_seeds`, `beef_cubes`, `cooking_oil`, `kangkong`, `sili_haba`, `sitaw`, `tilapia`, `tofu`.

## 3. Model Regression Diagnosis
### Controlled Comparison
Evaluated three checkpoints on a frozen "v1 validation subset" (331 images):

| Model | mAP50 | mAP50-95 | Verdict |
| :--- | :--- | :--- | :--- |
| YOLOv8n | 0.024 | 0.021 | Baseline / Best |
| YOLOv12n | 0.010 | 0.007 | Regression |
| YOLOv26n | 0.010 | 0.008 | Regression |

**Findings**: 
- Performance dropped by \~60% when moving from v8 to v12/v26.
- Architecture discrepancy: The thesis mentions YOLOv8, but training logs show YOLOv12.
- **Diagnosis**: Regression is likely due to domain shift from stock-photo data and unoptimized hyperparameters for newer architectures.

## 4. Supplementary Image Workflow
Created a workspace at `datasets/timpla_supplement/` including:
- **Priority List**: `collection_priorities.csv` targeting zero-val classes.
- **Researcher Guide**: `researcher_guide.md` providing capture and annotation requirements.
- **Provenance Schema**: `provenance_template.json` for tracking licenses and sources.
- **Merge Utility**: `ml/supplement_merge.py` for injecting new data into the active combined set.

## 5. Recipe Pipeline & Data Model
Implemented a a multi-modal ingestion system:
- **Website Collector**: Extracts JSON-LD (Schema.org) and raw HTML.
- **Social Collector**: Framework for YouTube/TikTok metadata acquisition.
- **Normalization Engine**: Maps raw ingredients to canonical IDs with confidence scores and alias handling.
- **Data Model**: Implemented in `recipes/schema.json`, distinguishing between original text and normalized IDs.

## 6. Validation & Testing
- **Test Suite**: `tests/test_pipeline.py` covers normalization, recipe processing, and box validation.
- **Fixed**: Resolved a critical bug in `RecipeNormalizer` where generic aliases overrode specific ones (e.g., "chili" vs "bird's eye chili").

## 7. Reproducible Commands

### Dataset Merge
```bash
export PYTHONPATH=$PYTHONPATH:/Users/natty/vscode/timpla
./timpla-env/bin/python3 ml/merge_datasets.py
```

### Model Evaluation
```bash
export PYTHONPATH=$PYTHONPATH:/Users/natty/vscode/timpla
./timpla-env/bin/python3 ml/eval.py ml/models/yolov8n.pt ml/models/yolo12n.pt
```

### Recipe Pipeline
```bash
export PYTHONPATH=$PYTHONPATH:/Users/natty/vscode/timpla
./timpla-env/bin/python3 recipes/pipeline_runner.py
```

## 8. High-Priority Next Actions
1. **Fill Validation Gaps**: Execute the supplementary collection guide for the 8 missing classes.
2. **Hyperparameter Tuning**: Re-tune YOLOv12/v26 specifically for the 66-class TIMPLA taxonomy.
3. **Recipe Review**: Manually review the `recipes/review/queue.json` to approve normalized mappings.
