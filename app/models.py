from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Ingredient(Base):
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    parent_ingredient_id: Mapped[Optional[int]] = mapped_column(ForeignKey("ingredients.id"))
    category: Mapped[Optional[str]] = mapped_column(String(80))
    specificity_level: Mapped[int] = mapped_column(Integer, default=0)

    parent: Mapped[Optional["Ingredient"]] = relationship(
        remote_side=[id], back_populates="children"
    )
    children: Mapped[list["Ingredient"]] = relationship(back_populates="parent")
    aliases: Mapped[list["IngredientAlias"]] = relationship(
        back_populates="ingredient", cascade="all, delete-orphan"
    )


class IngredientAlias(Base):
    __tablename__ = "ingredient_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), index=True)
    alias: Mapped[str] = mapped_column(String(120), index=True)
    language: Mapped[str] = mapped_column(String(20), default="en")

    ingredient: Mapped[Ingredient] = relationship(back_populates="aliases")


class Dish(Base):
    __tablename__ = "dishes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    category: Mapped[Optional[str]] = mapped_column(String(80))
    description: Mapped[Optional[str]] = mapped_column(Text)
    variants: Mapped[list["Variant"]] = relationship(
        back_populates="dish", cascade="all, delete-orphan"
    )


class Variant(Base):
    __tablename__ = "variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dish_id: Mapped[int] = mapped_column(ForeignKey("dishes.id"), index=True)
    variant_name: Mapped[str] = mapped_column(String(120))
    region: Mapped[Optional[str]] = mapped_column(String(80))
    minimum_required_pct: Mapped[float] = mapped_column(Float, default=0.0)

    dish: Mapped[Dish] = relationship(back_populates="variants")
    recipes: Mapped[list["Recipe"]] = relationship(
        back_populates="variant", cascade="all, delete-orphan"
    )


class Recipe(Base):
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    variant_id: Mapped[int] = mapped_column(ForeignKey("variants.id"), index=True)
    source_name: Mapped[Optional[str]] = mapped_column(String(160))
    source_url: Mapped[Optional[str]] = mapped_column(String(500))
    servings: Mapped[Optional[int]] = mapped_column(Integer)
    instructions: Mapped[str] = mapped_column(Text, default="")

    variant: Mapped[Variant] = relationship(back_populates="recipes")
    ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        back_populates="recipe", cascade="all, delete-orphan"
    )


class RecipeIngredient(Base):
    __tablename__ = "recipe_ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id"), index=True)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredients.id"), index=True)
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    amount: Mapped[Optional[str]] = mapped_column(String(80))
    unit: Mapped[Optional[str]] = mapped_column(String(40))
    alternative_group: Mapped[Optional[str]] = mapped_column(String(80))

    recipe: Mapped[Recipe] = relationship(back_populates="ingredients")
    ingredient: Mapped[Ingredient] = relationship()
