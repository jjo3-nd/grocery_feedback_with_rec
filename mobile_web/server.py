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
from Python import simulated_annealing_fixed_amount

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
]

DISPLAY_ORDER = [
    "DESCRIPTION",
    "wweia_food_category_description",
    "Level_2_Category",
    "Level_1_Category",
]

DEFAULT_PRIORITY = [
    "wweia_food_category_description",
    "DESCRIPTION",
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
        if key == "DESCRIPTION":
            if "FOODCODE" in matches.columns and "DESCRIPTION" in matches.columns:
                subset = matches[["FOODCODE", "DESCRIPTION"]].dropna()
                if not subset.empty:
                    row = subset.iloc[0]
                    return {
                        "column": "FOODCODE",
                        "value": str(row["FOODCODE"]).strip(),
                        "description": str(row["DESCRIPTION"]).strip(),
                        "label": str(row["DESCRIPTION"]).strip(),
                    }
        else:
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


def get_first_description_pair(matches: pd.DataFrame) -> dict | None:
    if "FOODCODE" not in matches.columns or "DESCRIPTION" not in matches.columns:
        return None
    subset = matches[["FOODCODE", "DESCRIPTION"]].dropna()
    subset = subset[
        (subset["FOODCODE"].astype(str).str.len() > 0)
        & (subset["DESCRIPTION"].astype(str).str.len() > 0)
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    return {
        "label": str(row["DESCRIPTION"]).strip(),
        "column": "FOODCODE",
        "value": str(row["FOODCODE"]).strip(),
        "description": str(row["DESCRIPTION"]).strip(),
    }


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
                    "options": [],
                    "default": None,
                }
            )
            continue

        matches = food_pool[pool_names.str.contains(product_name, case=False, na=False, regex=False)]
        options = []
        for col in DISPLAY_ORDER:
            if col == "DESCRIPTION":
                pair = get_first_description_pair(matches)
                if pair:
                    options.append(pair)
            else:
                value = get_first_value(matches, col)
                if value:
                    options.append({"label": value, "column": col, "value": value})

        default_choice = select_default_for_product(product_name, matches)

        options_payload.append(
            {
                "index": int(idx),
                "product_name": product_name,
                "options": options,
                "default": default_choice,
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
        description = default_choice.get("description", "")
        if column not in diet_df.columns:
            diet_df[column] = ""
        diet_df.at[idx, column] = value
        diet_df.at[idx, "Recommendation_Hierarchy"] = column
        if column == "FOODCODE" and description:
            if "DESCRIPTION" not in diet_df.columns:
                diet_df["DESCRIPTION"] = ""
            diet_df.at[idx, "DESCRIPTION"] = description
    return diet_df


def enrich_diet_with_pool(diet_df: pd.DataFrame, food_pool: pd.DataFrame) -> pd.DataFrame:
    if "Product Name" not in diet_df.columns or "Product Name" not in food_pool.columns:
        return diet_df

    pool = food_pool.drop_duplicates(subset=["Product Name"]).copy()
    merged = diet_df.merge(pool, on="Product Name", how="left", suffixes=("", "_pool"))

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

    # Remove any remaining pool columns
    extra_cols = [c for c in merged.columns if c.endswith("_pool")]
    if extra_cols:
        merged.drop(columns=extra_cols, inplace=True)

    # Fallback contains-match for rows with no pool data
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
            description = str(override.get("description", "")).strip()
            if not column or not value:
                continue
            if column not in diet_df.columns:
                diet_df[column] = ""
            diet_df.at[idx, column] = value
            diet_df.at[idx, "Recommendation_Hierarchy"] = column
            if column == "FOODCODE" and description:
                if "DESCRIPTION" not in diet_df.columns:
                    diet_df["DESCRIPTION"] = ""
                diet_df.at[idx, "DESCRIPTION"] = description

    niter = request.form.get("niter", str(NITER_DEFAULT))
    try:
        niter = int(niter)
    except ValueError:
        niter = NITER_DEFAULT

    use_openai = request.form.get("use_openai", "").lower() in {"1", "true", "yes"}
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        openai_key = getattr(python_module, "OPENAI_API_KEY", "")

    diet_df = enrich_diet_with_pool(diet_df, food_pool)
    diet_df = ensure_recommendation_hierarchy(diet_df, food_pool)

    if use_openai and not openai_key:
        return jsonify({"error": "OPENAI_API_KEY is not set."}), 400

    result = simulated_annealing_fixed_amount(
        diet_df,
        food_pool,
        niter=niter,
        use_openai=use_openai,
        openai_api_key=openai_key,
    )
    best_table = attach_images(result["best_diet_table"])

    csv_buffer = io.StringIO()
    best_table.to_csv(csv_buffer, index=False)

    return jsonify(
        {
            "original_score": result["original_score"],
            "recommended_score": result["recommended_score"],
            "original_components": result.get("original_components", {}),
            "recommended_components": result.get("recommended_components", {}),
            "recipe_info": result.get("recipe_info"),
            "recipe_data": result.get("recipe_data"),
            "rows": best_table.to_dict(orient="records"),
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
