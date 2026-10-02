import os, yaml
from collections import Counter

with open("timpla_combined/data.yaml") as f:
    classes = yaml.safe_load(f)["names"]

counts = Counter()
label_dir = "timpla_combined/train/labels"
for fname in os.listdir(label_dir):
    with open(os.path.join(label_dir, fname)) as f:
        for c in {int(line.split()[0]) for line in f if line.strip()}:
            counts[classes[c]] += 1

for name in classes:
    print(f"{name}: {counts.get(name, 0)}")