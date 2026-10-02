import os, random, shutil
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# Localfood (Filipino-specific, thin)
rf.workspace("localfood").project("object-detection-bounding-box-pj6iu") \
  .version(1).download("yolov8", location="datasets/localfood")

# FOOD-INGREDIENTS (generic proteins/aromatics, already split)
rf.workspace("food-recipe-ingredient-images-0gnku").project("food-ingredients-dataset") \
  .version(3).download("yolov8", location="datasets/food_ingredients")


def split_if_needed(root, val_frac=0.2, test_frac=0.1):
    """Localfood ships as a single train/ folder; carve out valid/ and test/."""
    if os.path.isdir(os.path.join(root, "valid", "images")):
        return
    random.seed(42)
    img_dir = os.path.join(root, "train", "images")
    lbl_dir = os.path.join(root, "train", "labels")
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    n_val, n_test = int(len(images) * val_frac), int(len(images) * test_frac)
    groups = {"valid": images[:n_val], "test": images[n_val:n_val + n_test]}
    for split, files in groups.items():
        os.makedirs(os.path.join(root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, split, "labels"), exist_ok=True)
        for f in files:
            stem = os.path.splitext(f)[0]
            shutil.move(os.path.join(img_dir, f), os.path.join(root, split, "images", f))
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl):
                shutil.move(lbl, os.path.join(root, split, "labels", stem + ".txt"))
    print(f"Split {root}: train {len(images)-n_val-n_test}, valid {n_val}, test {n_test}")


split_if_needed("datasets/localfood")