# Timpla API

FastAPI thesis prototype for Filipino ingredient recognition and recipe feasibility.

## Setup

```bash
source timpla-env/bin/activate
pip install -r requirements.txt
python -m app.seed_db
uvicorn app.main:app --reload
```

Standalone dataset and training tools live in [`ml/`](ml/). Generated datasets,
weights, runs, and training logs are intentionally ignored by git.

## Machine learning with Docker

Build the training image from the repository root:

```bash
docker build -f ml/Dockerfile -t timpla-ml .
docker run --rm -v "$PWD:/workspace" timpla-ml 10
```

The repository is mounted so the container can read `timpla_combined/` and
write training outputs. Pass model paths after the epoch count to train a
specific set, for example `docker run --rm -v "$PWD:/workspace" timpla-ml 10 ml/models/yolov8n.pt`.

The API uses `timpla.db` by default. Set `DATABASE_URL` to override it. The ontology is loaded from `timpla_combined/data.yaml`; its `names` list is the source of truth for `/ingredients` and seeding.

## Endpoints

- `POST /detect_ingredients`: multipart field `image`; runs the trained YOLO model.
- `POST /recommend_recipes`: JSON `{ "detected_ingredients": ["chicken_thigh", "vinegar"] }`.
- `GET /recipes/{recipe_id}`: full recipe detail.
- `GET /ingredients`: ontology and aliases.
- `GET /docs`: interactive OpenAPI documentation.

Specific ingredients satisfy their generic parent, but generic detections do not satisfy a specific recipe ingredient. Approved substitutions are intentionally not seeded until their `alternative_group` values are defined.
