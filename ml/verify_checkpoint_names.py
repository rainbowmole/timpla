import torch
from ultralytics import YOLO
import yaml
import sys

def get_checkpoint_names(path):
    try:
        model = YOLO(path)
        # In Ultralytics, model.names is the index-to-name dict
        return model.names
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_yaml_names(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        names = data.get('names', [])
        if isinstance(names, dict):
            # Ensure we get them in sorted key order
            return [names[k] for k in sorted(names.keys())]
        return names

current_yaml = "timpla_combined/data.yaml"
checkpoints = [
    "runs/detect/timpla_runs/ml/models/yolov8n_e50/weights/best.pt",
    "runs/detect/timpla_runs/ml/models/yolo12n_e50-2/weights/best.pt",
    "runs/detect/timpla_runs/yolo26n_e50/weights/best.pt"
]

current_names = load_yaml_names(current_yaml)
print(f"Current YAML Names ({len(current_names)} classes):")
print(current_names)
print("-" * 40)

for cp in checkpoints:
    print(f"\nCheckpoint: {cp}")
    cp_names_dict = get_checkpoint_names(cp)
    if cp_names_dict is None:
        continue
    
    cp_names = [cp_names_dict[i] for i in range(len(cp_names_dict))]
    print(f"Number of classes: {len(cp_names)}")
    
    if cp_names == current_names:
        print("MATCH: Class order matches current data.yaml")
    else:
        print("MISMATCH: Class order differs")
        # Find mapping
        mapping = {}
        for i, name in enumerate(cp_names):
            try:
                curr_idx = current_names.index(name)
                mapping[i] = curr_idx
            except ValueError:
                mapping[i] = f"UNKNOWN({name})"
        
        print("Mapping (Checkpoint Index -> Current Index):")
        print(mapping)
        
        # Show specific diffs
        for i, (cn, pn) in enumerate(zip(current_names, cp_names)):
            if i < len(cp_names) and cn != pn:
                print(f"Diff at index {i}: Current={cn}, Checkpoint={pn}")
