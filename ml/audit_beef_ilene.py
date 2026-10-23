import os
import yaml
from pathlib import Path
from collections import Counter

def main():
    src_dir = Path("ingredients-6")
    with open(src_dir / "data.yaml", 'r') as f:
        src_data = yaml.safe_load(f)
        src_names = src_data['names']
        if isinstance(src_names, dict):
            src_names = [src_names[k] for k in sorted(src_names.keys())]

    beef_keywords = ["beef"]
    beef_classes = [i for i, name in enumerate(src_names) if any(k in name.lower() for k in beef_keywords)]
    
    print(f"{'Source Class':<20} | {'ID':<5} | {'Tr-Img':<8} | {'Tr-Inst':<8} | {'Val-Img':<8} | {'Val-Inst':<8}")
    print("-" * 70)

    train_lbl_dir = src_dir / "train" / "labels"
    val_lbl_dir = src_dir / "valid" / "labels"
    
    for cid in beef_classes:
        name = src_names[cid]
        
        # Train counts
        t_imgs, t_inst = 0, 0
        if train_lbl_dir.exists():
            for lbl_file in train_lbl_dir.glob("*.txt"):
                with open(lbl_file, 'r') as f:
                    lines = f.read().splitlines()
                    if any(line.startswith(f"{cid} ") for line in lines):
                        t_imgs += 1
                    t_inst += sum(1 for line in lines if line.startswith(f"{cid} "))
        
        # Val counts
        v_imgs, v_inst = 0, 0
        if val_lbl_dir.exists():
            for lbl_file in val_lbl_dir.glob("*.txt"):
                with open(lbl_file, 'r') as f:
                    lines = f.read().splitlines()
                    if any(line.startswith(f"{cid} ") for line in lines):
                        v_imgs += 1
                    v_inst += sum(1 for line in lines if line.startswith(f"{cid} "))
        
        print(f"{name:<20} | {cid:<5} | {t_imgs:<8} | {t_inst:<8} | {v_imgs:<8} | {v_inst:<8}")
        
        # Sample image paths
        print("  Samples:")
        if train_lbl_dir.exists():
            count = 0
            for lbl_file in train_lbl_dir.glob("*.txt"):
                with open(lbl_file, 'r') as f:
                    if any(line.startswith(f"{cid} ") for line in f):
                        # Find image
                        stem = lbl_file.stem
                        for ext in ['.jpg', '.jpeg', '.png']:
                            img = src_dir / "train" / "images" / (stem + ext)
                            if img.exists():
                                print(f"    - {img}")
                                count += 1
                                break
                if count >= 3: break
        print("-" * 30)

if __name__ == "__main__":
    main()
