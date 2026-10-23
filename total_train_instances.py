import os
from collections import defaultdict

dataset_path = '/Users/natty/vscode/timpla/timpla_combined_v20261020_ilene_v6_fixed'
label_dir = os.path.join(dataset_path, 'train', 'labels')

total_instances = 0
class_counts = defaultdict(int)

for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'): continue
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            cls_id = int(parts[0])
            class_counts[cls_id] += 1
            total_instances += 1

print(f"Total training instances: {total_instances}")
# We already have the chili counts from previous runs.
