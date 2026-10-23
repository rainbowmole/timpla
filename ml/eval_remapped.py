import torch
from ultralytics import YOLO
import yaml
import sys
import numpy as np
from collections import defaultdict

def load_yaml_names(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', [])
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return names

def get_remapped_metrics(model_path, data_yaml, v1_list_path):
    # Load current canonical names
    current_names = load_yaml_names(data_yaml)
    
    # Load model and its specific names
    model = YOLO(model_path)
    model_names = model.names # {index: name}
    
    # Create mapping: Model Index -> Current Canonical Index
    remap = {}
    for m_idx, m_name in model_names.items():
        try:
            current_idx = current_names.index(m_name)
            remap[m_idx] = current_idx
        except ValueError:
            remap[m_idx] = -1 # Not in current set

    print(f"\nEvaluating {model_path}")
    print(f"Model classes: {len(model_names)} | Current classes: {len(current_names)}")
    
    # Use the model's own val method but we need to override the names for a consistent report
    # However, model.val() uses the data.yaml specified in the model or passed.
    # To do a TRUE remapped eval without modifying files, we have to run inference manually on the v1 set.
    
    # Load v1 images
    with open(v1_list_path, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]

    # Run inference
    results = model.predict(img_paths, conf=0.001, iou=0.6, imgsz=640, device="mps", verbose=False)
    
    # We need to calculate mAP. Since we can't easily use model.val() with a dynamic remap
    # without modifying the data.yaml, we'll implement a simplified mAP calculation or 
    # use a temporary data.yaml that matches the model's names.
    
    # Actually, the easiest way to verify if the remap FIXES the issue is to 
    # create a TEMP data.yaml that matches the model's names exactly, 
    # run model.val(), and then map the results back.
    
    temp_yaml = f"temp_{model_path.split('/')[-1]}.yaml"
    with open(temp_yaml, 'w') as f:
        f.write("train: {}\nval: {}\ntest: {}\n".format("dummy", "dummy", "dummy"))
        f.write(f"nc: {len(model_names)}\nnames:\n")
        for i in range(len(model_names)):
            f.write(f"- {model_names[i]}\n")
            
    # We need a temporary validation set that uses the model's indices
    # The original labels in timpla_combined/valid are in CURRENT index.
    # We must convert them to MODEL index.
    
    # [Wait, this is getting complex. Let's use a simpler approach: 
    # Create a temporary directory, copy the v1-subset labels, 
    # remap the IDs in the .txt files from CURRENT -> MODEL, 
    # then run model.val()]
    
    return temp_yaml

if __name__ == "__main__":
    # This script is a placeholder for the logic. 
    # I will implement the full re-evaluation in a separate bash block.
    pass
