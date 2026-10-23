import os

dataset_path = '/Users/natty/vscode/timpla/timpla_combined_v20261020_ilene_v6_fixed'
chili_classes = {27, 57, 58, 59}
others = {31: 'cucumber', 64: 'talong'}

mixed = []

for split in ['train', 'valid', 'test']:
    label_dir = os.path.join(dataset_path, split, 'labels')
    if not os.path.exists(label_dir): continue
    
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'): continue
        
        with open(os.path.join(label_dir, label_file), 'r') as f:
            classes = set()
            for line in f:
                parts = line.split()
                if not parts: continue
                classes.add(int(parts[0]))
        
        has_chili = any(c in chili_classes for c in classes)
        has_other = any(c in others for c in classes)
        
        if has_chili and has_other:
            mixed.append((split, label_file, classes))

for split, file, cls in mixed:
    print(f"{split} | {file} | {cls}")

