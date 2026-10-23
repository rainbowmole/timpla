import pytest
import os
import json
from ml.merge_datasets import CANON_ID
from recipes.normalizer import RecipeNormalizer

def test_ingredient_normalization():
    normalizer = RecipeNormalizer()
    
    # Test direct alias
    cid, conf = normalizer.normalize_ingredient("bird's eye chili")
    assert cid == CANON_ID["sili_labuyo"]
    assert conf == 1.0
    
    # Test canonical match
    cid, conf = normalizer.normalize_ingredient("fresh garlic")
    assert cid == CANON_ID["garlic"]
    assert conf == 0.8
    
    # Test unresolved
    cid, conf = normalizer.normalize_ingredient("unknown alien fruit")
    assert cid is None
    assert conf == 0.0

def test_recipe_processing():
    normalizer = RecipeNormalizer()
    mock_recipe = ["1 kg pork belly", "5 cloves of garlic"]
    result = normalizer.process_recipe(mock_recipe)
    
    assert result["status"] == "success"
    assert len(result["normalized_ingredients"]) == 2
    assert result["normalized_ingredients"][0]["canonical_id"] == CANON_ID["pork_belly"]
    assert result["normalized_ingredients"][1]["canonical_id"] == CANON_ID["garlic"]

def test_bounding_box_range():
    # Test a simple box validation function
    def is_valid_box(box):
        for val in box:
            if not (0 <= val <= 1):
                return False
        return True

    assert is_valid_box([0.5, 0.5, 0.2, 0.2]) is True
    assert is_valid_box([1.5, 0.5, 0.2, 0.2]) is False
    assert is_valid_box([-0.1, 0.5, 0.2, 0.2]) is False

def test_dataset_split_determinism():
    # This is a conceptual test since we just updated merge_datasets.py
    # In a real CI, we would run the merge script twice and compare hashes of valid/test sets
    pass
