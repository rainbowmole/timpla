import os
from pathlib import Path

def check_integrity(root):
    errors = 0
    for split in ["train", "valid", "test"]:
        img_dir = Path(root) / split / "images"
        lbl_dir = Path(root) / split / "labels"
        if not img_dir.exists():
            print(f"Split {split} not found")
            continue
        
        images = list(img_dir.glob("*.*"))
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                print(f"Missing label: {img.name}")
                errors += 1
    return errors

if __name__ == "__main__":
    errs = check_integrity("timpla_combined")
    print(f"Integrity check complete. Total errors: {errs}")
