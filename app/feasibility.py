from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .models import Dish, Ingredient, IngredientAlias, Recipe, RecipeIngredient, Variant


@dataclass
class Match:
    ingredient: Ingredient
    score: float


def normalize_labels(db: Session, labels: list[str]) -> dict[int, Ingredient]:
    normalized: dict[int, Ingredient] = {}
    for label in labels:
        value = label.strip().lower()
        ingredient = db.scalar(select(Ingredient).where(Ingredient.canonical_name == value))
        if ingredient is None:
            ingredient = db.scalar(select(Ingredient).join(IngredientAlias).where(IngredientAlias.alias == value))
        if ingredient is not None:
            normalized[ingredient.id] = ingredient
    return normalized


def parent_ids(ingredient: Ingredient) -> set[int]:
    ids = set()
    current = ingredient
    while current is not None:
        ids.add(current.id)
        current = current.parent
    return ids


def match_ingredient(required: Ingredient, detected: dict[int, Ingredient]) -> Match | None:
    exact = detected.get(required.id)
    if exact is not None:
        return Match(exact, 1.0)
    for detected_ingredient in detected.values():
        if required.id in parent_ids(detected_ingredient) and detected_ingredient.id != required.id:
            return Match(detected_ingredient, 0.9)
    return None


def status_for(score: float) -> str:
    if score >= 0.85:
        return "Highly Feasible"
    if score >= 0.65:
        return "Feasible with Minor Omissions"
    if score >= 0.40:
        return "Requires Substitutions"
    return "Not Feasible"


def recommend(db: Session, labels: list[str]) -> list[dict]:
    detected = normalize_labels(db, labels)
    if not detected:
        return []

    candidate_ids = set()
    candidate_recipes = db.scalars(
        select(Recipe).options(joinedload(Recipe.ingredients).joinedload(RecipeIngredient.ingredient))
    ).unique()
    for recipe in candidate_recipes:
        if any(match_ingredient(item.ingredient, detected) for item in recipe.ingredients):
            candidate_ids.add(recipe.id)

    recipes = db.scalars(
        select(Recipe)
        .where(Recipe.id.in_(candidate_ids))
        .options(joinedload(Recipe.variant).joinedload(Variant.dish), joinedload(Recipe.ingredients).joinedload(RecipeIngredient.ingredient))
    ).unique().all()
    results = []
    for recipe in recipes:
        required = [item for item in recipe.ingredients if item.required]
        optional = [item for item in recipe.ingredients if not item.required]
        matches = {item.id: match_ingredient(item.ingredient, detected) for item in required}
        denominator = len(required) or 1
        score = sum(match.score for match in matches.values() if match is not None) / denominator
        optional_detected = [item.ingredient.canonical_name for item in optional if match_ingredient(item.ingredient, detected)]
        results.append(
            {
                "dish": recipe.variant.dish.name,
                "variant": recipe.variant.variant_name,
                "feasibility_score": round(score, 4),
                "status": status_for(score),
                "matched_required": [item.ingredient.canonical_name for item, match in zip(required, matches.values()) if match is not None],
                "missing_required": [item.ingredient.canonical_name for item, match in zip(required, matches.values()) if match is None],
                "optional_detected": optional_detected,
            }
        )
    return sorted(results, key=lambda item: item["feasibility_score"], reverse=True)
