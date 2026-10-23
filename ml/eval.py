"""
Score a model on the ORIGINAL validation images only (no pantry / Thanh Huy Phan
images, and no image that contains a new-class label), so the result is
comparable with the earlier 58-class runs.

Usage (from project root):
    python ml/eval_v1_val.py runs/detect/timpla_runs/ml/models/yolo12n_e50-2/weights/best.pt
Writes timpla_combined/valid_v1.txt and data_v1val.yaml.
"""
import os
import sys
import yaml

ROOT = os.path.abspath("timpla_combined")
OLD_PREFIXES = ("lf_", "fi_", "oiv7_")
NEW_CLASSES = {
    "banana_ketchup", "canned_sardines", "canned_tuna", "coconut_milk_can",
    "fish_sauce_bottle", "knorr_cube", "pechay", "upo",
}


def build_v1_val():
    with open(os.path.join(ROOT, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    new_ids = {i for i, n in enumerate(names) if n in NEW_CLASSES}
    img_dir = os.path.join(ROOT, "valid", "images")
    lbl_dir = os.path.join(ROOT, "valid", "labels")

    keep = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.startswith(OLD_PREFIXES):
            continue
        lp = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
        ids = set()
        if os.path.exists(lp):
            with open(lp) as f:
                ids = {int(l.split()[0]) for l in f if l.strip()}
        if ids & new_ids:
            continue
        keep.append(os.path.join(img_dir, fn))

    txt = os.path.join(ROOT, "valid_v1.txt")
    with open(txt, "w") as f:
        f.write("\n".join(keep) + "\n")

    yaml_path = os.path.abspath("data_v1val.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump({
            "path": ROOT,
            "train": "train/images",
            "val": txt,
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }, f, sort_keys=False)
    print(f"v1 validation subset: {len(keep)} images -> {txt}")
    return yaml_path


if __name__ == "__main__":
    data_yaml = build_v1_val()
    if len(sys.argv) < 2:
        print("Pass one or more weights files to evaluate.")
        sys.exit(0)
    from ultralytics import YOLO

    for w in sys.argv[1:]:
        m = YOLO(w).val(data=data_yaml, device="mps", plots=False)
        print(f"{w}: mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")