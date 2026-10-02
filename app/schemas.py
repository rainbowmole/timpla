from typing import Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    ingredient_id: str
    label: str
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4)
    count: int


class DetectionResponse(BaseModel):
    detections: list[Detection]


class RecommendationRequest(BaseModel):
    detected_ingredients: list[str] = Field(min_length=1)


class Recommendation(BaseModel):
    dish: str
    variant: str
    feasibility_score: float
    status: str
    matched_required: list[str]
    missing_required: list[str]
    optional_detected: list[str]


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation]


class IngredientResponse(BaseModel):
    id: int
    canonical_name: str
    parent_ingredient_id: Optional[int]
    category: Optional[str]
    specificity_level: int
    aliases: list[str]


class RecipeIngredientResponse(BaseModel):
    ingredient: str
    required: bool
    amount: Optional[str]
    unit: Optional[str]
    alternative_group: Optional[str]


class RecipeResponse(BaseModel):
    id: int
    dish: str
    variant: str
    region: Optional[str]
    source_name: Optional[str]
    source_url: Optional[str]
    servings: Optional[int]
    ingredients: list[RecipeIngredientResponse]
    instructions: str
