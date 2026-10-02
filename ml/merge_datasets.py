import os, shutil, yaml, random

CANONICAL_CLASSES = sorted([
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
])
CANON_ID = {n: i for i, n in enumerate(CANONICAL_CLASSES)}

OUTPUT_ROOT = "timpla_combined"
SPLITS = ["train", "valid", "test"]

# ── Source 1: Localfood ───────────────────────────────────────────────
LOCALFOOD_ROOT = "datasets/localfood"
LOCALFOOD_OVERRIDES = {
    "chicken_tight": "chicken_thigh",   # dataset typo (both spellings covered)
    "chicken_tigh": "chicken_thigh",
    "white_raddish": "radish",
}

# ── Source 2: FOOD-INGREDIENTS ────────────────────────────────────────
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
}

# ── Source 3: Open Images V7 subset ───────────────────────────────────
OIV7_ROOT = "datasets/open_images_subset"
# trim OIV7_MAP to just the genuinely starved classes
OIV7_MAP = {
    "crab": "crab",
    "shrimp": "shrimp",
    "squid": "squid",
}


def names_of(root):
    with open(os.path.join(root, "data.yaml")) as f:
        return yaml.safe_load(f)["names"]


def build_map(raw_names, overrides, allow_identity):
    m = {}
    for i, n in enumerate(raw_names):
        if n in overrides:
            m[i] = overrides[n]
        elif allow_identity and n in CANON_ID:
            m[i] = n
        else:
            m[i] = None
    return m


def to_box(parts):
    """Return [xc, yc, w, h] from a 5-value box row or a polygon row."""
    vals = list(map(float, parts[1:]))
    if len(parts) == 5:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def process(root, src_map, prefix):
    """For Localfood / FOOD-INGREDIENTS: existing train/valid/test split,
    class index looked up per-row via src_map."""
    for split in SPLITS:
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        out_img = os.path.join(OUTPUT_ROOT, split, "images")
        out_lbl = os.path.join(OUTPUT_ROOT, split, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        kept = 0
        for lf in os.listdir(lbl_dir):
            if not lf.endswith(".txt"):
                continue
            new_lines = []
            with open(os.path.join(lbl_dir, lf)) as f:
                for line in f:
                    parts = line.split()
                    # valid: 5-value box row, or polygon row (class + x/y pairs = odd length >= 7)
                    if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 == 0):
                        continue
                    try:
                        canon = src_map.get(int(parts[0]))
                        if canon is None:
                            continue
                        xc, yc, w, h = to_box(parts)
                    except ValueError:
                        continue  # corrupted row (e.g. two numbers stuck together)
                    new_lines.append(f"{CANON_ID[canon]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            if not new_lines:
                continue
            stem = os.path.splitext(lf)[0]
            img = next((os.path.join(img_dir, stem + e) for e in (".jpg", ".jpeg", ".png")
                        if os.path.exists(os.path.join(img_dir, stem + e))), None)
            if img is None:
                continue
            new_stem = f"{prefix}_{stem}"
            shutil.copy(img, os.path.join(out_img, new_stem + os.path.splitext(img)[1]))
            with open(os.path.join(out_lbl, new_stem + ".txt"), "w") as f:
                f.write("\n".join(new_lines) + "\n")
            kept += 1
        print(f"  {prefix}/{split}: kept {kept}")


def process_oiv7(root, class_map, prefix, val_frac=0.15, test_frac=0.1):
    """Each subfolder of `root` is one class with a flat images/ + darknet/ pair.
    No existing split, so we split per class here to keep it balanced across
    train/valid/test."""
    random.seed(42)
    for folder_name, canon in class_map.items():
        if canon is None:
            continue
        img_dir = os.path.join(root, folder_name, "images")
        lbl_dir = os.path.join(root, folder_name, "darknet")
        if not os.path.isdir(img_dir):
            print(f"  [skip] no folder for '{folder_name}'")
            continue

        images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)
        n_val = int(len(images) * val_frac)
        n_test = int(len(images) * test_frac)
        splits = {
            "valid": images[:n_val],
            "test": images[n_val:n_val + n_test],
            "train": images[n_val + n_test:],
        }

        for split, files in splits.items():
            out_img = os.path.join(OUTPUT_ROOT, split, "images")
            out_lbl = os.path.join(OUTPUT_ROOT, split, "labels")
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)
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
                        # every box in this folder belongs to `canon`, regardless
                        # of the original class index in the file
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


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process(LOCALFOOD_ROOT, build_map(names_of(LOCALFOOD_ROOT), LOCALFOOD_OVERRIDES, True), "lf")
    process(FOODING_ROOT, build_map(names_of(FOODING_ROOT), FOODING_MAP, False), "fi")
    process_oiv7(OIV7_ROOT, OIV7_MAP, "oiv7")
    with open(os.path.join(OUTPUT_ROOT, "data.yaml"), "w") as f:
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write(f"nc: {len(CANONICAL_CLASSES)}\nnames:\n")
        f.writelines(f"- {n}\n" for n in CANONICAL_CLASSES)
    print(f"Done: {len(CANONICAL_CLASSES)} classes -> {OUTPUT_ROOT}/")