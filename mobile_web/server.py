from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException
import pandas as pd

# Allow importing Python.py from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
sys.path.append(str(REPO_ROOT))

import Python as python_module
from Python import simulated_annealing_fixed_amount, calculate_hei_scores_wrapper, extract_hei_components

APP_DIR = Path(__file__).resolve().parent
default_pool = REPO_ROOT / "data" / "walmart_hei_2020.csv"
default_pool_path = str(default_pool) if default_pool.exists() else "/Users/jeongwon/Downloads/walmart_hei_2020.csv"
FOOD_POOL_PATH = os.getenv("FOOD_POOL_PATH", default_pool_path)
NITER_DEFAULT = int(os.getenv("NITER_DEFAULT", "1000"))

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

_food_pool_df: pd.DataFrame | None = None
_image_url_map: dict[str, str] = {}
HIERARCHY_COLUMNS = [
    "Level_2_Category",
    "wweia_food_category_description",
    "FOODCODE",
    "Level_1_Category",
    "Keyword",
]

DISPLAY_ORDER = [
    "Keyword",
    "wweia_food_category_description",
    "Level_2_Category",
    "Level_1_Category",
]

DEFAULT_PRIORITY = [
    "wweia_food_category_description",
    "Keyword",
    "Level_2_Category",
    "Level_1_Category",
]


def load_food_pool() -> pd.DataFrame:
    global _food_pool_df, _image_url_map
    if _food_pool_df is not None:
        return _food_pool_df

    food_pool = pd.read_csv(FOOD_POOL_PATH)
    _food_pool_df = food_pool

    if "Product Name" in food_pool.columns and "Image URL" in food_pool.columns:
        image_df = food_pool[["Product Name", "Image URL"]].dropna()
        image_df = image_df[image_df["Image URL"].astype(str).str.len() > 0]
        _image_url_map = (
            image_df.groupby("Product Name")["Image URL"].first().to_dict()
        )
    else:
        _image_url_map = {}

    return _food_pool_df


def attach_images(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Original_Food" in df.columns:
        df["Original_Image_URL"] = df["Original_Food"].map(_image_url_map).fillna("")
    else:
        df["Original_Image_URL"] = ""

    if "New_Food" in df.columns:
        df["New_Image_URL"] = df["New_Food"].map(_image_url_map).fillna("")
    else:
        df["New_Image_URL"] = ""

    return df


def select_default_for_product(product_name: str, matches: pd.DataFrame) -> dict | None:
    for key in DEFAULT_PRIORITY:
        if key in matches.columns:
            value = (
                matches[key]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .head(1)
            )
            if not value.empty:
                return {
                    "column": key,
                    "value": value.iloc[0],
                    "description": "",
                    "label": value.iloc[0],
                }
    return None


def get_first_value(matches: pd.DataFrame, col: str) -> str | None:
    if col not in matches.columns:
        return None
    values = (
        matches[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    if values.empty:
        return None
    return values.iloc[0]


def build_category_options(diet_df: pd.DataFrame, food_pool: pd.DataFrame) -> list[dict]:
    options_payload = []
    pool_names = food_pool["Product Name"].astype(str) if "Product Name" in food_pool.columns else pd.Series([], dtype=str)

    for idx, row in diet_df.iterrows():
        product_name = str(row.get("Product Name", "")).strip()
        if not product_name:
            options_payload.append(
                {
                    "index": int(idx),
                    "product_name": "",
                    "values": {},
                    "default_index": 0,
                }
            )
            continue

        matches = food_pool[pool_names.str.contains(product_name, case=False, na=False, regex=False)]
        values = {}
        for col in DISPLAY_ORDER:
            values[col] = get_first_value(matches, col) or ""

        default_choice = select_default_for_product(product_name, matches)
        default_index = 0
        if default_choice and default_choice.get("column") in DISPLAY_ORDER:
            default_index = DISPLAY_ORDER.index(default_choice["column"])

        options_payload.append(
            {
                "index": int(idx),
                "product_name": product_name,
                "values": values,
                "default_index": default_index,
            }
        )

    return options_payload


def ensure_recommendation_hierarchy(diet_df: pd.DataFrame, food_pool: pd.DataFrame) -> pd.DataFrame:
    if "Recommendation_Hierarchy" not in diet_df.columns:
        diet_df["Recommendation_Hierarchy"] = ""

    pool_names = food_pool["Product Name"].astype(str) if "Product Name" in food_pool.columns else pd.Series([], dtype=str)

    for idx, row in diet_df.iterrows():
        if str(row.get("Recommendation_Hierarchy", "")).strip():
            continue
        product_name = str(row.get("Product Name", "")).strip()
        if not product_name:
            continue
        matches = food_pool[pool_names.str.contains(product_name, case=False, na=False, regex=False)]
        if matches.empty:
            continue
        default_choice = select_default_for_product(product_name, matches)
        if not default_choice:
            continue
        column = default_choice["column"]
        value = default_choice["value"]
        if column not in diet_df.columns:
            diet_df[column] = ""
        diet_df.at[idx, column] = value
        diet_df.at[idx, "Recommendation_Hierarchy"] = column
    return diet_df


def enrich_diet_with_pool(diet_df: pd.DataFrame, food_pool: pd.DataFrame) -> pd.DataFrame:
    if "Product Name" not in diet_df.columns or "Product Name" not in food_pool.columns:
        return diet_df

    pool = food_pool.drop_duplicates(subset=["Product Name"]).copy()
    merged = diet_df.copy()

    def normalize_foodcode(series: pd.Series) -> pd.Series:
        norm = series.astype(str).str.strip()
        norm = norm.str.replace(r"\.0$", "", regex=True)
        norm = norm.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        return norm

    diet_food_code_col = None
    for candidate in ["FOODCODE", "Food Code", "FoodCode"]:
        if candidate in merged.columns:
            diet_food_code_col = candidate
            break

    pool_food_code_col = "FOODCODE" if "FOODCODE" in pool.columns else None

    if diet_food_code_col and pool_food_code_col:
        diet_codes = normalize_foodcode(merged[diet_food_code_col])
        pool_codes = normalize_foodcode(pool[pool_food_code_col])
        merged["_foodcode_key"] = diet_codes
        pool["_foodcode_key"] = pool_codes

        merged = merged.merge(pool, on="_foodcode_key", how="left", suffixes=("", "_pool"))
        merged.drop(columns=["_foodcode_key"], inplace=True, errors="ignore")
    else:
        merged = merged.merge(pool, on="Product Name", how="left", suffixes=("", "_pool"))

    pool_cols = [c for c in pool.columns if c != "Product Name"]
    for col in pool_cols:
        pool_col = f"{col}_pool"
        if pool_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = merged[col].where(~merged[col].isna(), merged[pool_col])
        else:
            merged[col] = merged[pool_col]
        merged.drop(columns=[pool_col], inplace=True)

    extra_cols = [c for c in merged.columns if c.endswith("_pool")]
    if extra_cols:
        merged.drop(columns=extra_cols, inplace=True)

    rep_col = None
    for candidate in ["Energy", "KCAL", "Calories"]:
        if candidate in pool_cols:
            rep_col = candidate
            break

    if rep_col and rep_col in merged.columns:
        unmatched = merged[rep_col].isna()
        if unmatched.any():
            pool_names = pool["Product Name"].astype(str)
            for idx, row in merged[unmatched].iterrows():
                name = str(row.get("Product Name", "")).strip()
                if not name:
                    continue
                matches = pool[pool_names.str.contains(name, case=False, na=False, regex=False)]
                if matches.empty:
                    continue
                match_row = matches.iloc[0]
                for col in pool_cols:
                    if col in merged.columns and pd.isna(merged.at[idx, col]):
                        merged.at[idx, col] = match_row[col]

    return merged


def row_components(row_df: pd.DataFrame) -> dict:
    scores = calculate_hei_scores_wrapper(row_df)
    return extract_hei_components(scores)


def compute_swap_totals_components(
    base_df: pd.DataFrame, replace_row: pd.Series, idx: int
) -> tuple[float, float, dict]:
    temp = base_df.copy()
    for col in base_df.columns:
        if col in replace_row.index:
            temp.at[idx, col] = replace_row[col]
    total_cost = temp["Price"].sum() if "Price" in temp.columns else 0.0
    scores = calculate_hei_scores_wrapper(temp)
    components = extract_hei_components(scores)
    total_score = components.get("HEI2015_TOTAL_SCORE", scores[-1] if len(scores) else 0)
    return total_cost, total_score, components


@app.get("/")
def index():
    return send_from_directory(str(APP_DIR), "index.html")


@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        return jsonify({"error": error.description}), error.code
    return jsonify({"error": f"Server error: {error}"}), 500


@app.post("/api/recommend")
def recommend():
    food_pool = load_food_pool()

    if "diet" not in request.files:
        return jsonify({"error": "Missing diet file upload under 'diet'."}), 400

    file = request.files["diet"]
    if not file.filename:
        return jsonify({"error": "Uploaded file has no name."}), 400

    try:
        diet_df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Failed to read CSV: {exc}"}), 400

    overrides_raw = request.form.get("overrides", "")
    if overrides_raw:
        try:
            overrides = json.loads(overrides_raw)
        except json.JSONDecodeError:
            overrides = []

        for override in overrides:
            idx = int(override.get("index"))
            column = str(override.get("column", "")).strip()
            value = str(override.get("value", "")).strip()
            if not column or not value:
                continue
            if column not in diet_df.columns:
                diet_df[column] = ""
            diet_df.at[idx, column] = value
            diet_df.at[idx, "Recommendation_Hierarchy"] = column

    niter = request.form.get("niter", str(NITER_DEFAULT))
    try:
        niter = int(niter)
    except ValueError:
        niter = NITER_DEFAULT

    use_openai = request.form.get("use_openai", "").lower() in {"1", "true", "yes"}
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        openai_key = getattr(python_module, "OPENAI_API_KEY", "")

    if use_openai and not openai_key:
        return jsonify({"error": "OPENAI_API_KEY is not set."}), 400

    diet_df = enrich_diet_with_pool(diet_df, food_pool)
    diet_df = ensure_recommendation_hierarchy(diet_df, food_pool)

    bundles = python_module.get_all_three_bundles(
        diet_df,
        food_pool,
        niter=niter,
        use_openai=use_openai,
        openai_api_key=openai_key,
    )

    def prep_full(bundle):
        df = bundle["best_diet_full"].copy()
        df = df.iloc[: len(diet_df)].reset_index(drop=True)
        return df

    health_full = prep_full(bundles["healthiest"])
    cheap_full = prep_full(bundles["cheapest"])
    balanced_full = prep_full(bundles["balanced"])

    base_cost = diet_df["Price"].sum() if "Price" in diet_df.columns else 0.0
    base_components = extract_hei_components(calculate_hei_scores_wrapper(diet_df))
    base_score = base_components.get("HEI2015_TOTAL_SCORE", 0)

    rows = []
    for idx in range(len(diet_df)):
        orig_row = diet_df.iloc[idx]
        orig_name = str(orig_row.get("Product Name", ""))
        orig_price = float(orig_row.get("Price", 0.0)) if "Price" in diet_df.columns else 0.0
        orig_image = _image_url_map.get(orig_name, "")
        orig_components = row_components(orig_row.to_frame().T)

        def build_option(option_row: pd.Series):
            name = str(option_row.get("Product Name", ""))
            price = float(option_row.get("Price", 0.0)) if "Price" in option_row.index else 0.0
            image = _image_url_map.get(name, "")
            components = row_components(option_row.to_frame().T)
            total_cost, total_score, total_components = compute_swap_totals_components(diet_df, option_row, idx)
            return {
                "name": name,
                "price": price,
                "image": image,
                "components": components,
                "hei": components.get("HEI2015_TOTAL_SCORE", 0),
                "total_cost_after": total_cost,
                "total_score_after": total_score,
                "total_components_after": total_components,
            }

        target_column = orig_row.get("Recommendation_Hierarchy", "")
        target_value = orig_row.get(target_column, "") if target_column in orig_row else ""

        row_data = {
            "target_category": target_value,
            "target_category_column": target_column,
            "original": {
                "name": orig_name,
                "price": orig_price,
                "image": orig_image,
                "components": orig_components,
                "hei": orig_components.get("HEI2015_TOTAL_SCORE", 0),
            },
            "options": {
                "healthiest": build_option(health_full.iloc[idx]),
                "cheapest": build_option(cheap_full.iloc[idx]),
                "balanced": build_option(balanced_full.iloc[idx]),
            },
        }
        rows.append(row_data)

    csv_buffer = io.StringIO()
    pd.DataFrame(
        {
            "Original_Food": [r["original"]["name"] for r in rows],
            "Original_Price": [r["original"]["price"] for r in rows],
            "Target_Category": [r["target_category"] for r in rows],
            "Target_Category_Column": [r.get("target_category_column", "") for r in rows],
            "Healthiest_Food": [r["options"]["healthiest"]["name"] for r in rows],
            "Healthiest_Price": [r["options"]["healthiest"]["price"] for r in rows],
            "Cheapest_Food": [r["options"]["cheapest"]["name"] for r in rows],
            "Cheapest_Price": [r["options"]["cheapest"]["price"] for r in rows],
            "Balanced_Food": [r["options"]["balanced"]["name"] for r in rows],
            "Balanced_Price": [r["options"]["balanced"]["price"] for r in rows],
        }
    ).to_csv(csv_buffer, index=False)

    return jsonify(
        {
            "summary": {
                "original_score": base_score,
                "original_cost": base_cost,
                "healthiest_score": bundles["healthiest"].get("recommended_score"),
                "cheapest_cost": bundles["cheapest"].get("recommended_cost"),
                "balanced_score": bundles["balanced"].get("recommended_score"),
            },
            "base_components": base_components,
            "original_components": bundles["healthiest"].get("original_components", {}),
            "recommended_components": bundles["healthiest"].get("recommended_components", {}),
            "recipe_info": bundles["healthiest"].get("recipe_info"),
            "recipe_data": bundles["healthiest"].get("recipe_data"),
            "rows": rows,
            "csv": csv_buffer.getvalue(),
        }
    )


@app.post("/api/options")
def options():
    food_pool = load_food_pool()

    if "diet" not in request.files:
        return jsonify({"error": "Missing diet file upload under 'diet'."}), 400

    file = request.files["diet"]
    if not file.filename:
        return jsonify({"error": "Uploaded file has no name."}), 400

    try:
        diet_df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Failed to read CSV: {exc}"}), 400

    options_payload = build_category_options(diet_df, food_pool)
    return jsonify({"items": options_payload})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
