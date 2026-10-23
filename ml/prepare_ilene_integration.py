import os
import yaml
import json
import csv
import hashlib
import shutil
from pathlib import Path
from collections import Counter

def get_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    src_dir = Path("ingredients-6")
    canon_dir = Path("timpla_combined").resolve() # resolve symlink
    dry_run_dir = Path("timpla_combined_ilene_v6_dryrun")
    
    with open("canonical_list.txt", "r") as f:
        canonical_classes = [line.strip() for line in f if line.strip()]
    canon_id = {name: i for i, name in enumerate(canonical_classes)}

    with open(src_dir / "data.yaml", 'r') as f:
        src_data = yaml.safe_load(f)
        src_names = src_data['names']
        if isinstance(src_names, dict):
            src_names = [src_names[k] for k in sorted(src_names.keys())]

    mapping_rules = {
        "ampalaya": "ampalaya", "bangus": "bangus", "bay leaves": "bay_leaves",
        "beef": "beef", "beef intestines": "beef", "beef kidney": "beef",
        "beef ribs": "beef", "beef tail": "beef", "beef tripe": "beef",
        "bokchoy": "pechay", "breadcrumbs": "exclude", "broccoli": "exclude",
        "brown sugar": "exclude", "butter": "butter", "cabbage": "cabbage",
        "calamansi": "calamansi", "canola oil": "cooking_oil", "cashew nuts": "exclude",
        "cauliflower": "exclude", "cellophane noodles": "exclude", "chayote": "sayote",
        "chicharon": "exclude", "chicken": "chicken_generic", "chicken bouillon granules": "exclude",
        "chickpeas": "exclude", "chorizo": "exclude", "cloves": "exclude", "corn": "exclude",
        "dried shitake mushrooms": "mushroom", "egg": "egg", "eggplant": "talong",
        "garlic": "garlic", "ginger": "ginger", "green beans": "sitaw",
        "ham": "pork_generic", "hotdog": "exclude", "kangkong": "kangkong",
        "lemon": "exclude", "mackerel": "exclude", "malungay leaves": "malunggay",
        "mung beans": "exclude", "okra": "okra", "olive": "exclude", "onion": "onion",
        "pata": "pork_belly", "pork heart": "pork_generic", "pork liver": "pork_generic",
        "pork loin": "pork_generic", "pork lung": "pork_generic", "pork meat": "pork_generic",
        "pork ribs": "pork_generic", "pork skin": "pork_generic", "potato": "potato",
        "puso ng saging": "exclude", "raisins": "exclude", "red bell pepper": "bell_pepper",
        "red chili pepper": "sili_labuyo", "rice": "rice_grains", "saba banana": "exclude",
        "sesame seeds": "exclude", "singkamas": "radish", "sitaw": "sitaw",
        "spring onions": "spring_onions", "tomato": "tomato"
    }

    mapping_csv_path = "ilene_mapping.csv"
    with open(mapping_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["source_class", "source_class_id", "action", "canonical_class", "confidence", "reason", "train_images", "train_instances", "valid_images", "valid_instances"])
        
        train_lbl_dir = src_dir / "train" / "labels"
        class_counts = Counter()
        image_counts = Counter()
        
        if train_lbl_dir.exists():
            for lbl_file in train_lbl_dir.glob("*.txt"):
                with open(lbl_file, 'r') as f:
                    lines = f.read().splitlines()
                    classes_in_img = set()
                    for line in lines:
                        if not line.strip(): continue
                        cls_id = int(line.split()[0])
                        class_counts[cls_id] += 1
                        classes_in_img.add(cls_id)
                    for c in classes_in_img:
                        image_counts[c] += 1

        for i, name in enumerate(src_names):
            action = mapping_rules.get(name, "exclude")
            canon = mapping_rules[name] if action == "map_to_existing" else "N/A"
            conf = "1.0" if action == "map_to_existing" else "0.0"
            reason = "Matched to canonical" if action == "map_to_existing" else "Not in taxonomy"
            writer.writerow([name, i, action, canon, conf, reason, image_counts[i], class_counts[i], 0, 0])

    dry_run_img_dir = dry_run_dir / "train" / "images"
    dry_run_lbl_dir = dry_run_dir / "train" / "labels"
    dry_run_img_dir.mkdir(parents=True, exist_ok=True)
    dry_run_lbl_dir.mkdir(parents=True, exist_ok=True)

    combined_hashes = set()
    if canon_dir.exists():
        for img_path in canon_dir.rglob("*"):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                combined_hashes.add(get_sha256(img_path))

    added_counts = Counter()
    added_instances = Counter()
    class_image_counts = Counter()
    CAP = 500

    train_img_dir = src_dir / "train" / "images"
    train_lbl_dir = src_dir / "train" / "labels"

    for lbl_file in sorted(train_lbl_dir.glob("*.txt")):
        img_stem = lbl_file.stem
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            p = train_img_dir / (img_stem + ext)
            if p.exists():
                img_file = p
                break
        
        if img_file is None: continue
        if get_sha256(img_file) in combined_hashes:
            continue
            
        new_lines = []
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                src_id = int(parts[0])
                src_name = src_names[src_id]
                action = mapping_rules.get(src_name, "exclude")
                if action == "map_to_existing":
                    canon_name = mapping_rules[src_name]
                    if class_image_counts[canon_name] >= CAP:
                        continue
                    new_lines.append(f"{canon_id[canon_name]} {' '.join(parts[1:])}")
                    added_instances[canon_name] += 1
        
        if new_lines:
            shutil.copy2(img_file, dry_run_img_dir / img_file.name)
            with open(dry_run_lbl_dir / lbl_file.name, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            
            # Update cap for the first valid class found in image
            first_line = open(lbl_file).readline().split()
            if first_line:
                first_cls_name = src_names[int(first_line[0])]
                if first_cls_name in mapping_rules and mapping_rules[first_cls_name] != "exclude":
                    class_image_counts[mapping_rules[first_cls_name]] += 1
            
            # Record added images per class
            for line in new_lines:
                cid = int(line.split()[0])
                added_counts[canonical_classes[cid]] += 1

    with open(dry_run_dir / "data.yaml", 'w') as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(canonical_classes)}\nnames:\n")
        for i, name in enumerate(canonical_classes):
            f.write(f"- {name}\n")

    print("--- Dry Run Summary ---")
    for cls in sorted(canonical_classes):
        if added_counts[cls] > 0:
            print(f"{cls}: {added_counts[cls]} instances added")
    
    print(f"\nTotal images added: {len(list(dry_run_img_dir.glob('*')))}")
    print(f"Total instances added: {sum(added_instances.values())}")

if __name__ == "__main__":
    main()
