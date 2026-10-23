import yaml
from pathlib import Path

src_dir = Path("ingredients-6")
with open(src_dir / "data.yaml", 'r') as f:
    data = yaml.safe_load(f)
    src_names = data['names']
    if isinstance(src_names, dict):
        src_names = [src_names[k] for k in sorted(src_names.keys())]

mapping_rules = {
    "ampalaya": "ampalaya",
    "bangus": "bangus",
    "bay leaves": "bay_leaves",
    "beef": "beef",
    "beef intestines": "beef",
    "beef kidney": "beef",
    "beef ribs": "beef",
    "beef tail": "beef",
    "beef tripe": "beef",
    "bokchoy": "pechay",
    "breadcrumbs": "exclude",
    "broccoli": "exclude",
    "brown sugar": "exclude",
    "butter": "butter",
    "cabbage": "cabbage",
    "calamansi": "calamansi",
    "canola oil": "cooking_oil",
    "cashew nuts": "exclude",
    "cauliflower": "exclude",
    "cellophane noodles": "exclude",
    "chayote": "sayote",
    "chicharon": "exclude",
    "chicken": "chicken_generic",
    "chicken bouillon granules": "exclude",
    "chickpeas": "exclude",
    "chorizo": "exclude",
    "cloves": "exclude",
    "corn": "exclude",
    "dried shitake mushrooms": "mushroom",
    "egg": "egg",
    "eggplant": "talong",
    "garlic": "garlic",
    "ginger": "ginger",
    "green beans": "sitaw",
    "ham": "pork_generic",
    "hotdog": "exclude",
    "kangkong": "kangkong",
    "lemon": "exclude",
    "mackerel": "exclude",
    "malungay leaves": "malunggay",
    "mung beans": "exclude",
    "okra": "okra",
    "olive": "exclude",
    "onion": "onion",
    "pata": "pork_belly",
    "pork heart": "pork_generic",
    "pork liver": "pork_generic",
    "pork loin": "pork_generic",
    "pork lung": "pork_generic",
    "pork meat": "pork_generic",
    "pork ribs": "pork_generic",
    "pork skin": "pork_generic",
    "potato": "potato",
    "puso ng saging": "exclude",
    "raisins": "exclude",
    "red bell pepper": "bell_pepper",
    "red chili pepper": "sili_labuyo",
    "rice": "rice_grains",
    "saba banana": "exclude",
    "sesame seeds": "exclude",
    "singkamas": "radish",
    "sitaw": "sitaw",
    "spring onions": "spring_onions",
    "tomato": "tomato"
}

print("Checking for mapping mismatches...")
for name in src_names:
    if name not in mapping_rules:
        print(f"MISSING: '{name}' is not in mapping_rules")
