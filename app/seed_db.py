from pathlib import Path

import yaml
from sqlalchemy import select

from .database import Base, SessionLocal, engine
from .models import Dish, Ingredient, IngredientAlias, Recipe, RecipeIngredient, Variant


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_names() -> list[str]:
    with (ROOT_DIR / "timpla_combined/data.yaml").open() as file:
        return yaml.safe_load(file)["names"]


def seed() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.scalar(select(Ingredient.id).limit(1)) is not None:
            print("Database already contains data; skipping seed.")
            return
        ingredients = {}
        for name in load_names():
            ingredients[name] = Ingredient(canonical_name=name, specificity_level=1)
            db.add(ingredients[name])
        db.flush()
        for generic, specifics in {"chicken_generic": ["chicken_breast", "chicken_thigh", "chicken_leg", "chicken_wing"], "pork_generic": ["pork_belly"]}.items():
            for specific in specifics:
                ingredients[specific].parent = ingredients[generic]
                ingredients[specific].specificity_level = 2
        for alias, name in {"chicken": "chicken_generic", "soy": "soy_sauce", "patis": "fish_sauce"}.items():
            if name in ingredients:
                db.add(IngredientAlias(alias=alias, ingredient=ingredients[name], language="en"))

        dishes = [
            ("Adobo", "stew", "A savory Filipino braise.", [("Chicken Adobo", "Luzon", ["chicken_generic", "soy_sauce", "vinegar", "garlic", "bay_leaves", "pepper_corn"], ["onion"]), ("Pork Adobo", "Luzon", ["pork_generic", "soy_sauce", "vinegar", "garlic", "bay_leaves", "pepper_corn"], ["onion"])]),
            ("Sinigang", "soup", "A sour tamarind-based soup.", [("Pork Sinigang", "Tagalog", ["pork_belly", "tomato", "onion", "radish", "sili", "tamarind"], ["kangkong", "okra"]), ("Shrimp Sinigang", "Tagalog", ["shrimp", "tomato", "onion", "radish", "sili"], ["kangkong", "okra"])]),
            ("Tinola", "soup", "A light ginger chicken soup.", [("Chicken Tinola", "Visayas", ["chicken_generic", "ginger", "onion", "sayote"], ["malunggay", "lemon_grass"]), ("Chicken Tinola with Papaya", "Luzon", ["chicken_generic", "ginger", "onion", "papaya"], ["malunggay", "lemon_grass"])]),
            ("Ginisang Gulay", "vegetable", "Sauteed mixed vegetables.", [("Ginisang Gulay", "Nationwide", ["garlic", "onion", "tomato", "cooking_oil"], ["sitaw", "carrot", "cabbage"]), ("Ginisang Gulay with Tofu", "Nationwide", ["tofu", "garlic", "onion", "tomato", "cooking_oil"], ["sitaw", "carrot", "cabbage"])]),
        ]
        for dish_name, category, description, variants in dishes:
            dish = Dish(name=dish_name, category=category, description=description)
            db.add(dish)
            for variant_name, region, required, optional in variants:
                variant = Variant(dish=dish, variant_name=variant_name, region=region, minimum_required_pct=0.65)
                db.add(variant)
                db.flush()
                recipe = Recipe(variant=variant, source_name="Timpla starter recipes", servings=4, instructions="Prepare ingredients, cook until tender, and season to taste.")
                db.add(recipe)
                for name in required:
                    if name in ingredients:
                        db.add(RecipeIngredient(recipe=recipe, ingredient=ingredients[name], required=True))
                for name in optional:
                    if name in ingredients:
                        db.add(RecipeIngredient(recipe=recipe, ingredient=ingredients[name], required=False))
        db.commit()
        print("Seeded Timpla database.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
