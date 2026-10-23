"""
Timpla dataset merge (v2).

Sources
  1. Localfood            datasets/localfood            (Filipino-specific, thin)
  2. FOOD-INGREDIENTS v3  datasets/food_ingredients     (+ bottle gourd -> upo)
  3. Open Images V7       datasets/open_images_subset   (crab, shrimp, squid ONLY)
  4. pantry-wlooe         datasets/pantry               (branded / canned goods)
  5. Thanh Huy Phan       datasets/thp_ingredients      (starved protein / veg classes)

Run from the project root:   python ml/merge_datasets.py
Rebuilds timpla_combined/ from scratch every time (it is a generated folder).
"""
import os
import re
import shutil
import random
import yaml
import json
from datetime import datetime

# ── Canonical classes ────────────────────────────────────────────────
BASE_CLASSES = [
    "ampalaya", "annatto_seeds", "bangus", "bay_leaves", "beef", "beef_cubes",
    "bell_pepper", "bulaw", "butter", "cabbage", "calamansi", "carrot",
    "chicken_breast", "chicken_leg", "chicken_thigh", "chicken_wing",
    "chicken_generic", "chili", "cooking_oil", "crab", "cucumber", "egg",
    "gabi", "garlic", "ginger", "kalabasa", "kangkong", "lemon_grass",
    "malunggay", "mushroom", "okra", "onion", "papaya", "pepper_corn",
    "pork_belly", "pork_generic", "potato", "radish", "red_onion",
    "rice_grains", "salt", "sayote", "shrimp", "sili", "sili_haba",
    "sili_labuyo", "sitaw", "soy_sauce", "spring_onions", "squid",
    "talong", "taro_leaves", "tilapia", "tofu", "tomato", "turmeric_powder",
    "vinegar", "white_onion",
]
NEW_CLASSES = {
    "banana_ketchup", "canned_sardines", "canned_tuna", "coconut_milk_can",
    "fish_sauce_bottle", "knorr_cube", "pechay", "upo",
}
CANONICAL_CLASSES = sorted(set(BASE_CLASSES) | NEW_CLASSES)
CANON_ID = {n: i for i, n in enumerate(CANONICAL_CLASSES)}

OUTPUT_ROOT = "timpla_combined"
# VERSIONED_ROOT = f"timpla_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SPLITS = ["train", "valid", "test"]
SPLIT_ALIASES = {"train": ["train"], "valid": ["valid", "val"], "test": ["test"]}
IMG_EXTS = (".jpg", ".jpeg", ".png")

# ── Source 1: Localfood (identity mapping + fixes) ───────────────────
LOCALFOOD_ROOT = "datasets/localfood"
LOCALFOOD_OVERRIDES = {
    "chicken_tight": "chicken_thigh",
    "chicken_tigh": "chicken_thigh",
    "white_raddish": "radish",
}

# ── Source 2: FOOD-INGREDIENTS v3 ────────────────────────────────────
FOODING_ROOT = "datasets/food_ingredients"
FOODING_MAP = {
    "Garlic": "garlic", "Onion": "onion", "Onion Leaves": "spring_onions",
    "Ginger": "ginger", "Chicken": "chicken_generic", "Pork": "pork_generic",
    "Beef": "beef", "Egg": "egg", "Chili Pepper -Khursani-": "chili",
    "Soy Sauce": "soy_sauce", "Tomato": "tomato", "Potato": "potato",
    "Carrot": "carrot", "Cabbage": "cabbage", "Cucumber": "cucumber",
    "Capsicum": "bell_pepper", "Butter": "butter", "Salt": "salt",
    "Mushroom": "mushroom", "Crab Meat": "crab", "Radish": "radish",
    "Papaya": "papaya", "Moringa Leaves -Sajyun ko Munta-": "malunggay",
    "Taro Root-Pidalu-": "gabi", "Taro Leaves -Karkalo-": "taro_leaves",
    "Bitter Gourd": "ampalaya", "Pumpkin -Farsi-": "kalabasa",
    "Long Beans -Bodi-": "sitaw", "Chayote-iskus-": "sayote",
    "Brinjal": "talong", "Green Brinjal": "talong", "Okra -Bhindi-": "okra",
    "Rice -Chamal-": "rice_grains", "Tofu": "tofu",
    # new: upo is a standard sinigang vegetable
    "Bottle Gourd -Lauka-": "upo",
    # Add these only when a dish in your recipe set needs them (each new class
    # needs recipes, validation images and UI labels), then add the class name
    # to BASE_CLASSES above:
    #   "Sponge Gourd -Ghiraula-": "patola",
    #   "Jack Fruit": "langka",
    #   "Bamboo Shoots -Tama-": "labong",
    #   "Sweet Potato -Suthuni-": "kamote",
}

# ── Source 3: Open Images V7 (one folder per class, no splits) ───────
# Only the genuinely starved classes: blanket use caused domain-shift losses.
OIV7_ROOT = "datasets/open_images_subset"
OIV7_MAP = {"crab": "crab", "shrimp": "shrimp", "squid": "squid"}

# ── Source 4: pantry-wlooe (Roboflow, branded / canned goods) ────────
PANTRY_ROOT = "datasets/pantry"
PANTRY_MAP = {
    "Knorr-Seasoning": "knorr_cube",
    "Jufran": "banana_ketchup",
    "Fish-Sauce": "fish_sauce_bottle",
    "Sardines": "canned_sardines",
    "Canned-Tuna": "canned_tuna",
    "Coconut-Milk": "coconut_milk_can",
}

# ── Source 5: Thanh Huy Phan "food-ingredient-recognition" ───────────
# Stock-photo style, so only classes that are starved everywhere else.
# Keys are matched ignoring case, spaces, hyphens and underscores.
# A WARN line for a key that does not exist is harmless: it prints the
# dataset's real class names so you can fix the key.
THP_ROOT = "datasets/thp_ingredients"
THP_MAP = {
    "chicken breast": "chicken_breast",
    "chicken wing": "chicken_wing",
    "pork belly": "pork_belly",
    "bok choy": "pechay",
    # These didn't exist in data.yaml, so they were warnings.
    # Looking at actual names, we have 'chicken breast' and 'chicken wing'
    # but no 'chicken thigh' or 'chicken leg'.
}


# ── Helpers ──────────────────────────────────────────────────────────
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def names_of(root):
    with open(os.path.join(root, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def build_map(raw_names, mapping, identity=False, label="", warn=True):
    """source class id -> canonical name (or None). Warns about unmatched keys."""
    nm = {norm(k): v for k, v in mapping.items()}
    out, seen = {}, set()
    for i, n in enumerate(raw_names):
        k = norm(n)
        if k in nm:
            out[i] = nm[k]
            seen.add(k)
        elif identity and n in CANON_ID:
            out[i] = n
            seen.add(k)
        else:
            out[i] = None
            if warn:
                # Log labels found in source but not in mapping
                with open("dropped_labels.log", "a") as log:
                    log.write(f"[{label}] Dropped label: {n} (norm: {k})\n")
    missing = sorted(k for k in nm if k not in seen)
    if missing and warn:
        print(f"  [WARN {label}] mapping keys not found in data.yaml: {missing}")
        print(f"  [WARN {label}] actual class names: {list(raw_names)}")
    return out


def check_targets():
    for label, m in [("localfood", LOCALFOOD_OVERRIDES), ("fooding", FOODING_MAP),
                     ("oiv7", OIV7_MAP), ("pantry", PANTRY_MAP), ("thp", THP_MAP)]:
        bad = [v for v in m.values() if v not in CANON_ID]
        assert not bad, f"{label} maps to unknown canonical classes: {bad}"


def to_box(parts):
    """[xc, yc, w, h] from a 5-value box row or a polygon row."""
    vals = list(map(float, parts[1:]))
    if len(parts) == 5:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def find_image(img_dir, stem):
    for e in IMG_EXTS:
        p = os.path.join(img_dir, stem + e)
        if os.path.exists(p):
            return p
    return None


def out_dirs(split, root=None):
    target_root = root if root else OUTPUT_ROOT
    img = os.path.join(target_root, split, "images")
    lbl = os.path.join(target_root, split, "labels")
    os.makedirs(img, exist_ok=True)
    os.makedirs(lbl, exist_ok=True)
    return img, lbl


# ── Roboflow-style sources (train/valid/test + data.yaml) ────────────
def process(root, src_map, prefix, target_root=None):
    for out_split in SPLITS:
        src_split = next((s for s in SPLIT_ALIASES[out_split]
                          if os.path.isdir(os.path.join(root, s, "images"))), None)
        if src_split is None:
            continue
        img_dir = os.path.join(root, src_split, "images")
        lbl_dir = os.path.join(root, src_split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        out_img, out_lbl = out_dirs(out_split, root=target_root)
        kept = 0
        for lf in sorted(os.listdir(lbl_dir)):
            if not lf.endswith(".txt"):
                continue
            new_lines = []
            with open(os.path.join(lbl_dir, lf)) as f:
                for line in f:
                    parts = line.split()
                    # valid: 5-value box row, or polygon row (class + x/y pairs)
                    if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 == 0):
                        continue
                    try:
                        canon = src_map.get(int(parts[0]))
                        if canon is None:
                            continue
                        xc, yc, w, h = to_box(parts)
                    except ValueError:
                        continue  # corrupted row
                    new_lines.append(f"{CANON_ID[canon]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            if not new_lines:
                continue
            stem = os.path.splitext(lf)[0]
            img = find_image(img_dir, stem)
            if img is None:
                continue
            new_stem = f"{prefix}_{stem}"
            shutil.copy(img, os.path.join(out_img, new_stem + os.path.splitext(img)[1]))
            with open(os.path.join(out_lbl, new_stem + ".txt"), "w") as f:
                f.write("\n".join(new_lines) + "\n")
            kept += 1
        print(f"  {prefix}/{out_split}: kept {kept}")


# ── Open Images (per-class folders, own split) ───────────────────────
def process_oiv7(root, class_map, prefix, target_root=None, val_frac=0.15, test_frac=0.1):
    # random.seed(42)  # No longer needed if we use sorted and fixed indices
    for folder_name, canon in class_map.items():
        img_dir = os.path.join(root, folder_name, "images")
        lbl_dir = os.path.join(root, folder_name, "darknet")
        if not os.path.isdir(img_dir):
            print(f"  [skip] no folder for '{folder_name}'")
            continue

        # DETERMINISTIC ORDERING: sorted(os.listdir) is consistent across runs
        images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))

        n_val, n_test = int(len(images) * val_frac), int(len(images) * test_frac)
        splits = {"valid": images[:n_val],
                  "test": images[n_val:n_val + n_test],
                  "train": images[n_val + n_test:]}

        # PERSISTED MANIFESTS
        manifest_path = os.path.join(root, f"manifest_{folder_name}.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                saved_splits = json.load(f)
                # Ensure all images are still present
                current_images = set(images)
                if all(set(v).issubset(current_images) for v in saved_splits.values()):
                    splits = saved_splits
                    print(f"  oiv7/{folder_name}: using persisted manifest")
                else:
                    print(f"  oiv7/{folder_name}: manifest mismatch, regenerating")
        else:
            # Save the generated splits for reproducibility
            with open(manifest_path, "w") as f:
                json.dump(splits, f)

        for split, files in splits.items():
            out_img, out_lbl = out_dirs(split, root=target_root)
            kept = 0
            for img_file in files:
                stem = os.path.splitext(img_file)[0]
                lbl_path = os.path.join(lbl_dir, stem + ".txt")
                if not os.path.exists(lbl_path):
                    continue
                new_lines = []
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        try:
                            xc, yc, w, h = map(float, parts[1:])
                        except ValueError:
                            continue
                        new_lines.append(f"{CANON_ID[canon]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                if not new_lines:
                    continue
                new_stem = f"{prefix}_{canon}_{stem}"
                shutil.copy(os.path.join(img_dir, img_file),
                            os.path.join(out_img, new_stem + os.path.splitext(img_file)[1]))
                with open(os.path.join(out_lbl, new_stem + ".txt"), "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                kept += 1
            print(f"  oiv7/{canon}/{split}: kept {kept}")


# ── Reporting and safety net ─────────────────────────────────────────
def class_ids_in(path):
    with open(path) as f:
        return {int(l.split()[0]) for l in f if l.strip()}


def split_counts(root=None):
    """{split: {class_id: number of images containing that class}}"""
    target_root = root if root else OUTPUT_ROOT
    counts = {s: {} for s in SPLITS}
    for s in SPLITS:
        d = os.path.join(target_root, s, "labels")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            for c in class_ids_in(os.path.join(d, fn)):
                counts[s][c] = counts[s].get(c, 0) + 1
    return counts


def ensure_val_coverage(root=None, min_train=3):
    """New classes only: if a new class has no validation image, move one train
    image into valid. Old classes are never touched, so the original validation
    set stays comparable with earlier results."""
    target_root = root if root else OUTPUT_ROOT
    counts = split_counts(root=target_root)
    new_ids = {CANON_ID[n] for n in NEW_CLASSES}
    lbl_dir = os.path.join(target_root, "train", "labels")
    img_dir = os.path.join(target_root, "train", "images")
    for name in sorted(NEW_CLASSES):
        cid = CANON_ID[name]
        n_val, n_train = counts["valid"].get(cid, 0), counts["train"].get(cid, 0)
        if n_val > 0:
            continue
        if n_train < min_train:
            print(f"  [WARN] {name}: no valid images and only {n_train} train - needs more data")
            continue
        candidates = []
        for fn in sorted(os.listdir(lbl_dir)):
            ids = class_ids_in(os.path.join(lbl_dir, fn))
            if cid in ids:
                candidates.append((0 if ids <= new_ids else 1, fn))
        candidates.sort()  # prefer images that contain only new classes
        fn = candidates[0][1]
        stem = os.path.splitext(fn)[0]
        img = find_image(img_dir, stem)
        v_img, v_lbl = out_dirs("valid", root=target_root)
        shutil.move(os.path.join(lbl_dir, fn), os.path.join(v_lbl, fn))
        shutil.move(img, os.path.join(v_img, os.path.basename(img)))
        print(f"  [fix] {name}: moved 1 train image to valid (source had none)")


def report(root=None):
    target_root = root if root else OUTPUT_ROOT
    counts = split_counts(root=target_root)
    print("\nNew classes (images per split):")
    print(f"  {'class':<20}{'train':>7}{'valid':>7}{'test':>7}")
    for name in sorted(NEW_CLASSES):
        c = CANON_ID[name]
        print(f"  {name:<20}{counts['train'].get(c, 0):>7}{counts['valid'].get(c, 0):>7}{counts['test'].get(c, 0):>7}")
    no_val = [n for n in CANONICAL_CLASSES if counts["valid"].get(CANON_ID[n], 0) == 0]
    print(f"\nClasses with NO validation image (cannot be scored): {no_val}")
    totals = {s: len(os.listdir(os.path.join(target_root, s, 'images')))
              for s in SPLITS if os.path.isdir(os.path.join(target_root, s, 'images'))}
    print(f"Total images: {totals}")


if __name__ == "__main__":
    # Versioning: use timestamp for output root to preserve history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_root = f"{OUTPUT_ROOT}_{timestamp}"

    check_targets()

    if os.path.isdir(versioned_root):
        shutil.rmtree(versioned_root)
    os.makedirs(versioned_root)

    roboflow_sources = [
        ("localfood", LOCALFOOD_ROOT, LOCALFOOD_OVERRIDES, True, "lf", False),
        ("food_ingredients", FOODING_ROOT, FOODING_MAP, False, "fi", True),
        ("pantry", PANTRY_ROOT, PANTRY_MAP, False, "pt", True),
        ("thp", THP_ROOT, THP_MAP, False, "thp", True),
    ]
    for label, root, mapping, identity, prefix, warn in roboflow_sources:
        if not os.path.isdir(root):
            print(f"[skip] {label}: {root} not found")
            continue
        print(f"Processing {label}...")
        process(root, build_map(names_of(root), mapping, identity, label, warn), prefix, target_root=versioned_root)

    if os.path.isdir(OIV7_ROOT):
        print("Processing open images...")
        process_oiv7(OIV7_ROOT, OIV7_MAP, "oiv7", target_root=versioned_root)
    else:
        print(f"[skip] open images: {OIV7_ROOT} not found")

    with open(os.path.join(versioned_root, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(CANONICAL_CLASSES)}\nnames:\n")
        f.writelines(f"- {n}\n" for n in CANONICAL_CLASSES)

    ensure_val_coverage(root=versioned_root)
    report(root=versioned_root)

    # Create symlink to make current version the "active" one
    if os.path.exists(OUTPUT_ROOT):
        if os.path.islink(OUTPUT_ROOT):
            os.unlink(OUTPUT_ROOT)
        else:
            shutil.rmtree(OUTPUT_ROOT)

    os.symlink(versioned_root, OUTPUT_ROOT)
    print(f"\nDone: {len(CANONICAL_CLASSES)} classes -> {versioned_root}/")
    print(f"Symlinked {OUTPUT_ROOT} -> {versioned_root}")