from contextlib import asynccontextmanager

import yaml
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .database import Base, engine, get_db
from .detection import IngredientDetector
from .feasibility import recommend
from .models import Ingredient, Recipe, RecipeIngredient, Variant
from .schemas import (
    DetectionResponse,
    IngredientResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecipeResponse,
    RecipeIngredientResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Timpla API", version="0.1.0", lifespan=lifespan)
detector = IngredientDetector()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect_ingredients", response_model=DetectionResponse)
async def detect_ingredients(image: UploadFile = File(...)):
    try:
        return {"detections": detector.predict(await image.read())}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommend_recipes", response_model=RecommendationResponse)
def recommend_recipes(request: RecommendationRequest, db: Session = Depends(get_db)):
    return {"recommendations": recommend(db, request.detected_ingredients)}


@app.get("/recipes/{recipe_id}", response_model=RecipeResponse)
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.scalar(
        select(Recipe)
        .where(Recipe.id == recipe_id)
        .options(joinedload(Recipe.variant).joinedload(Variant.dish), joinedload(Recipe.ingredients).joinedload(RecipeIngredient.ingredient))
    )
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {
        "id": recipe.id,
        "dish": recipe.variant.dish.name,
        "variant": recipe.variant.variant_name,
        "region": recipe.variant.region,
        "source_name": recipe.source_name,
        "source_url": recipe.source_url,
        "servings": recipe.servings,
        "instructions": recipe.instructions,
        "ingredients": [
            RecipeIngredientResponse(
                ingredient=item.ingredient.canonical_name,
                required=item.required,
                amount=item.amount,
                unit=item.unit,
                alternative_group=item.alternative_group,
            )
            for item in recipe.ingredients
        ],
    }


@app.get("/ingredients", response_model=list[IngredientResponse])
def get_ingredients(db: Session = Depends(get_db)):
    ontology_path = __import__("pathlib").Path(__file__).resolve().parent.parent / "timpla_combined/data.yaml"
    with ontology_path.open() as file:
        ontology_names = yaml.safe_load(file).get("names", [])
    ingredients = db.scalars(select(Ingredient).options(joinedload(Ingredient.aliases))).unique().all()
    by_name = {ingredient.canonical_name: ingredient for ingredient in ingredients}
    return [
        IngredientResponse(
            id=ingredient.id,
            canonical_name=ingredient.canonical_name,
            parent_ingredient_id=ingredient.parent_ingredient_id,
            category=ingredient.category,
            specificity_level=ingredient.specificity_level,
            aliases=[alias.alias for alias in ingredient.aliases],
        )
        for name in ontology_names
        if (ingredient := by_name.get(name)) is not None
    ]
