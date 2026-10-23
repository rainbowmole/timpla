import os
import sys
import yaml
from collections import defaultdict

def count_instances(dataset_path, target_classes_indices):
    """
    dataset_path: Path to the dataset root (containing train/labels)
    target_classes_indices: Dict of {class_name: index}
    """
    instance_counts = defaultdict(int)
    image_counts = defaultdict(set)
    
    labels_dir = os.path.join(dataset_path, 'train', 'labels')
    if not os.path.exists(labels_dir):
        print(f"Labels dir not found: {labels_dir}")
        return instance_counts, image_counts

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        img_name = label_file.replace('.txt', '')
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                
                # Reverse lookup for name
                class_name = next((name for name, idx in target_classes_indices.items() if idx == class_id), None)
                
                if class_name:
                    instance_counts[class_name] += 1
                    image_counts[class_name].add(img_name)
                    
    return instance_counts, image_counts

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python count_instances.py <dataset_path> <data_yaml_path>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    data_yaml_path = sys.argv[2]
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        names = data['names']
        # Create name -> index mapping
        name_to_idx = {name: i for i, name in enumerate(names)}
        
    counts, images = count_instances(dataset_path, name_to_idx)
    
    # Target classes to report
    targets = ['sili', 'sili_haba', 'sili_labuyo', 'chili', 'talong', 'cucumber']
    
    print(f"Dataset: {dataset_path}")
    for t in targets:
        print(f"{t}: instances={counts[t]}, images={len(images[t])}")

