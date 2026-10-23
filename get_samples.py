import os
from collections import defaultdict

dataset_path = '/Users/natty/vscode/timpla/timpla_combined_v20261020_ilene_v6_fixed'
target_classes = {
    27: 'chili',
    57: 'sili',
    58: 'sili_haba',
    59: 'sili_labuyo'
}

samples = defaultdict(list)

for split in ['train', 'valid', 'test']:
    label_dir = os.path.join(dataset_path, split, 'labels')
    if not os.path.exists(label_dir): continue
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                cls_id = int(parts[0])
                if cls_id in target_classes:
                    cls_name = target_classes[cls_id]
                    if len(samples[cls_name]) < 3:
                        # Use the image path
                        img_name = label_file.replace('.txt', '.jpg')
                        # Need to check if it's .jpg or .png or .jpeg
                        # For simplicity, I'll just list the label file and the likely image path
                        samples[cls_name].append(os.path.join(dataset_path, split, 'images', img_name))

for cls_name, paths in samples.items():
    print(f"{cls_name}:")
    for p in paths:
        print(f"  {p}")

