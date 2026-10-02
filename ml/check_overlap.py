import csv
import urllib.request

CANONICAL = {
    "ampalaya","annatto_seeds","bangus","bay_leaves","beef","beef_cubes",
    "bell_pepper","bulaw","butter","cabbage","calamansi","carrot",
    "chicken_breast","chicken_leg","chicken_thigh","chicken_wing",
    "chicken_generic","chili","cooking_oil","crab","cucumber","egg",
    "gabi","garlic","ginger","kalabasa","kangkong","lemon_grass",
    "malunggay","mushroom","okra","onion","papaya","pepper_corn",
    "pork_belly","pork_generic","potato","radish","red_onion",
    "rice_grains","salt","sayote","shrimp","sili","sili_haba",
    "sili_labuyo","sitaw","soy_sauce","spring_onions","squid",
    "talong","taro_leaves","tilapia","tofu","tomato","turmeric_powder",
    "vinegar","white_onion",
}

url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
rows = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

matches = []
for row in csv.reader(rows):
    if len(row) < 2:
        continue
    oid, name = row[0], row[1]
    key = name.lower().replace(" (food)", "").replace(" ", "_")
    if key in CANONICAL or any(c in name.lower() for c in CANONICAL if len(c) > 3):
        matches.append(name)

print(f"{len(matches)} candidate matches:")
for m in sorted(matches):
    print(" ", m)