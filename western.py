import fiftyone as fo
import fiftyone.zoo as foz

# Classes available in Open Images V7 that map to Timpla's canonical set
CLASS_MAP = {
    "Apple": "apple",  # example only — swap in your actual overlapping classes
    "Banana": "banana",
    "Tomato": "tomato",
    "Potato": "potato",
    "Cucumber": "cucumber",
    "Mushroom": "mushroom",
    "Egg (Food)": "egg",
    "Pumpkin": "kalabasa",
    "Garden Asparagus": "asparagus",
    "Carrot": "carrot",
}

TARGET_PER_CLASS = 150  # pick the number — this is what gives you consistency

merged = None
for oiv7_name in CLASS_MAP:
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[oiv7_name],
        max_samples=TARGET_PER_CLASS,
        only_matching=True,   # keep only boxes for this class, ignore others in the same image
        dataset_name=f"oiv7_{oiv7_name.replace(' ', '_')}",
    )
    merged = ds if merged is None else merged.merge(ds)

merged.export(
    export_dir="datasets/open_images_subset",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
)