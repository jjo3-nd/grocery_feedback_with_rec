# Grocery Feedback with Recommendations

Mobile-first web app that lets users upload a cart CSV, choose category constraints per item, and generate HEI-optimized product swaps. Optional OpenAI mode adds a recipe + missing ingredients.

## Quick start

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r mobile_web/requirements.txt

# 3) (Optional) create a .env file in the repo root
# FOOD_POOL_PATH=./data/walmart_hei_2020.csv
# OPENAI_API_KEY=your_openai_key

# 4) Run the server
python mobile_web/server.py
```

Open the app at:
```
http://localhost:5000
```

## Sample data included

- `data/dummy_data2.csv` (example cart upload)
- `data/walmart_hei_2020.csv` (food pool, ~35MB)

If you don't set `FOOD_POOL_PATH`, the server defaults to `data/walmart_hei_2020.csv` when it exists.

## Data requirements

### Food pool (required)
`walmart_hei_2020.csv` must include at least:
- `Product Name`
- Category fields used for filtering (e.g., `Level_1_Category`, `Level_2_Category`, `wweia_food_category_description`, `FOODCODE`, `DESCRIPTION`)
- HEI nutrient columns (used to calculate scores)
- Optional: `Image URL` for product thumbnails

### User upload CSV (cart)
The upload file must include:
- `Product Name`

Optional (if you have them):
- `Price`
- `Product Size Values`
- `Product Size Units`
- `Walmart Item ID`

Example header (from `data/dummy_data2.csv`):
```
Product Name,Walmart Item ID,Product Size Values,Product Size Units,Price,Food Code,Serving Size,Amount
```

If any HEI nutrient columns are missing, the server attempts to enrich the cart rows by matching `Product Name` with the food pool.

## How it works

1. Upload a cart CSV.
2. For each item, select a category value (e.g., Dairy or Milk). This determines which column is used for filtering:
   - Dairy -> `Level_1_Category`
   - Milk -> `Level_2_Category`
3. Click **Run recommendations** to get swaps.
4. Click **Run with recipe** to add OpenAI-based recipe + missing ingredients.

## Environment variables

- `FOOD_POOL_PATH` (optional): absolute path to `walmart_hei_2020.csv`
- `OPENAI_API_KEY` (optional): required for **Run with recipe**
- `NITER_DEFAULT` (optional): simulated annealing iterations (default 1000)

## Troubleshooting

- **"OPENAI_API_KEY is not set"**: set the environment variable and restart the server.
- **Missing or empty recommendations**: ensure `Product Name` matches the food pool. If names differ, improve naming consistency or update the matching logic in `mobile_web/server.py`.

