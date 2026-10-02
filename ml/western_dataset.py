from openimages.download import download_dataset

CLASSES = [
    "Bell pepper", "Cabbage", "Carrot", "Crab", "Cucumber",
    "Egg (Food)", "Mushroom", "Potato", "Radish", "Shrimp",
    "Squid", "Tomato",
]

if __name__ == "__main__":
    download_dataset(
        "datasets/open_images_subset",
        CLASSES,
        annotation_format="darknet",
        limit=150,
    )