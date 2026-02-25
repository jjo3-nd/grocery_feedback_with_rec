import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import math
from openai import OpenAI
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def hei_2020(
    kcal=0,
    vtotalleg=0,
    vdrkgrleg=0,
    f_total=0,
    fwholefrt=0,
    g_whole=0,
    d_total=0,
    pfallprotleg=0,
    pfseaplantleg=0,
    monopoly=0,
    satfat=0,
    sodium=0,
    g_refined=0,
    add_sugars=0,
):
    VEGDEN = 0
    if kcal > 0:
        VEGDEN = vtotalleg / (kcal / 1000)
    HEI2015C1_TOTALVEG = 5 * (VEGDEN / 1.1)
    if HEI2015C1_TOTALVEG > 5:
        HEI2015C1_TOTALVEG = 5
    if VEGDEN == 0:
        HEI2015C1_TOTALVEG = 0

    GRBNDEN = 0
    if kcal > 0:
        GRBNDEN = vdrkgrleg / (kcal / 1000)
    HEI2015C2_GREEN_AND_BEAN = 5 * (GRBNDEN / 0.2)
    if HEI2015C2_GREEN_AND_BEAN > 5:
        HEI2015C2_GREEN_AND_BEAN = 5
    if GRBNDEN == 0:
        HEI2015C2_GREEN_AND_BEAN = 0

    FRTDEN = 0
    if kcal > 0:
        FRTDEN = f_total / (kcal / 1000)
    HEI2015C3_TOTALFRUIT = 5 * (FRTDEN / 0.8)
    if HEI2015C3_TOTALFRUIT > 5:
        HEI2015C3_TOTALFRUIT = 5
    if FRTDEN == 0:
        HEI2015C3_TOTALFRUIT = 0

    WHFRDEN = 0
    if kcal > 0:
        WHFRDEN = fwholefrt / (kcal / 1000)
    HEI2015C4_WHOLEFRUIT = 5 * (WHFRDEN / 0.4)
    if HEI2015C4_WHOLEFRUIT > 5:
        HEI2015C4_WHOLEFRUIT = 5
    if WHFRDEN == 0:
        HEI2015C4_WHOLEFRUIT = 0

    WGRNDEN = 0
    if kcal > 0:
        WGRNDEN = g_whole / (kcal / 1000)
    HEI2015C5_WHOLEGRAIN = 10 * (WGRNDEN / 1.5)
    if HEI2015C5_WHOLEGRAIN > 10:
        HEI2015C5_WHOLEGRAIN = 10
    if WGRNDEN == 0:
        HEI2015C5_WHOLEGRAIN = 0

    DAIRYDEN = 0
    if kcal > 0:
        DAIRYDEN = d_total / (kcal / 1000)
    HEI2015C6_TOTALDAIRY = 10 * (DAIRYDEN / 1.3)
    if HEI2015C6_TOTALDAIRY > 10:
        HEI2015C6_TOTALDAIRY = 10
    if DAIRYDEN == 0:
        HEI2015C6_TOTALDAIRY = 0

    PROTDEN = 0
    if kcal > 0:
        PROTDEN = pfallprotleg / (kcal / 1000)
    HEI2015C7_TOTPROT = 5 * (PROTDEN / 2.5)
    if HEI2015C7_TOTPROT > 5:
        HEI2015C7_TOTPROT = 5
    if PROTDEN == 0:
        HEI2015C7_TOTPROT = 0

    SEAPLDEN = 0
    if kcal > 0:
        SEAPLDEN = pfseaplantleg / (kcal / 1000)
    HEI2015C8_SEAPLANT_PROT = 5 * (SEAPLDEN / 0.8)
    if HEI2015C8_SEAPLANT_PROT > 5:
        HEI2015C8_SEAPLANT_PROT = 5
    if SEAPLDEN == 0:
        HEI2015C8_SEAPLANT_PROT = 0

    FARATIO = 0
    HEI2015C9_FATTYACID = 0
    FARMIN = 1.2
    FARMAX = 2.5
    if satfat > 0:
        FARATIO = monopoly / satfat
    if satfat == 0 and monopoly == 0:
        HEI2015C9_FATTYACID = 0
    elif satfat == 0 and monopoly > 0:
        HEI2015C9_FATTYACID = 10
    elif FARATIO >= FARMAX:
        HEI2015C9_FATTYACID = 10
    elif FARATIO <= FARMIN:
        HEI2015C9_FATTYACID = 0
    else:
        HEI2015C9_FATTYACID = 10 * ((FARATIO - FARMIN) / (FARMAX - FARMIN))

    SODDEN = 0
    SODMIN = 1.1
    SODMAX = 2.0
    if kcal > 0:
        SODDEN = sodium / kcal
    HEI2015C10_SODIUM = 0
    if SODDEN <= SODMIN:
        HEI2015C10_SODIUM = 10
    elif SODDEN >= SODMAX:
        HEI2015C10_SODIUM = 0
    else:
        HEI2015C10_SODIUM = 10 - (10 * (SODDEN - SODMIN) / (SODMAX - SODMIN))

    RGDEN = 0
    RGMIN = 1.8
    RGMAX = 4.3
    if kcal > 0:
        RGDEN = g_refined / (kcal / 1000)
    HEI2015C11_REFINEDGRAIN = 0
    if RGDEN <= RGMIN:
        HEI2015C11_REFINEDGRAIN = 10
    elif RGDEN >= RGMAX:
        HEI2015C11_REFINEDGRAIN = 0
    else:
        HEI2015C11_REFINEDGRAIN = 10 - (10 * (RGDEN - RGMIN) / (RGMAX - RGMIN))

    SFAT_PERC = 0
    if kcal > 0:
        SFAT_PERC = 100 * (satfat * 9 / kcal)
    SFATMIN = 8
    SFATMAX = 16
    HEI2015C12_SFAT = 0
    if SFAT_PERC >= SFATMAX:
        HEI2015C12_SFAT = 0
    elif SFAT_PERC <= SFATMIN:
        HEI2015C12_SFAT = 10
    else:
        HEI2015C12_SFAT = 10 - (10 * (SFAT_PERC - SFATMIN) / (SFATMAX - SFATMIN))

    ADDSUG_PERC = 0
    if kcal > 0:
        ADDSUG_PERC = 100 * (add_sugars * 16 / kcal)
    ADDSUGMIN = 6.5
    ADDSUGMAX = 26
    HEI2015C13_ADDSUG = 0
    if ADDSUG_PERC >= ADDSUGMAX:
        HEI2015C13_ADDSUG = 0
    elif ADDSUG_PERC <= ADDSUGMIN:
        HEI2015C13_ADDSUG = 10
    else:
        HEI2015C13_ADDSUG = 10 - (
            10 * (ADDSUG_PERC - ADDSUGMIN) / (ADDSUGMAX - ADDSUGMIN)
        )

    if kcal == 0:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    HEI2015_TOTAL_SCORE = (
        HEI2015C1_TOTALVEG
        + HEI2015C2_GREEN_AND_BEAN
        + HEI2015C3_TOTALFRUIT
        + HEI2015C4_WHOLEFRUIT
        + HEI2015C5_WHOLEGRAIN
        + HEI2015C6_TOTALDAIRY
        + HEI2015C7_TOTPROT
        + HEI2015C8_SEAPLANT_PROT
        + HEI2015C9_FATTYACID
        + HEI2015C10_SODIUM
        + HEI2015C11_REFINEDGRAIN
        + HEI2015C12_SFAT
        + HEI2015C13_ADDSUG
    )

    return (
        HEI2015C1_TOTALVEG,
        HEI2015C2_GREEN_AND_BEAN,
        HEI2015C3_TOTALFRUIT,
        HEI2015C4_WHOLEFRUIT,
        HEI2015C5_WHOLEGRAIN,
        HEI2015C6_TOTALDAIRY,
        HEI2015C7_TOTPROT,
        HEI2015C8_SEAPLANT_PROT,
        HEI2015C9_FATTYACID,
        HEI2015C10_SODIUM,
        HEI2015C11_REFINEDGRAIN,
        HEI2015C12_SFAT,
        HEI2015C13_ADDSUG,
        HEI2015_TOTAL_SCORE,
    )


def calculate_hei_scores_wrapper(df):
    # Energy / KCAL
    kcal = (
        df["Energy"].sum()
        if "Energy" in df
        else (df["KCAL"].sum() if "KCAL" in df else 0)
    )

    # Saturated Fat (SFAT or SOLID_FATS)
    satfat = (
        df["SOLID_FATS"].sum()
        if "SOLID_FATS" in df
        else (df["SFAT"].sum() if "SFAT" in df else 0)
    )

    # Monounsaturated Fat
    mfat = (
        df["Fatty_Acids_Total_Monounsaturated"].sum()
        if "Fatty_Acids_Total_Monounsaturated" in df
        else (df["MFAT"].sum() if "MFAT" in df else 0)
    )

    # Polyunsaturated Fat
    pfat = (
        df["Fatty_Acids_Total_Polyunsaturated"].sum()
        if "Fatty_Acids_Total_Polyunsaturated" in df
        else (df["PFAT"].sum() if "PFAT" in df else 0)
    )

    monopoly = mfat + pfat

    def get_sum(col):
        return df[col].sum() if col in df else 0

    f_total = get_sum("F_TOTAL")
    fwholefrt = get_sum("F_CITMLB") + get_sum("F_OTHER")
    vtotalleg = get_sum("V_TOTAL") + get_sum("V_LEGUMES")
    vdrkgrleg = get_sum("V_DRKGR") + get_sum("V_LEGUMES")

    pfallprotleg = (
        get_sum("PF_MPS_TOTAL")
        + get_sum("PF_EGGS")
        + get_sum("PF_NUTSDS")
        + get_sum("PF_SOY")
        + get_sum("PF_LEGUMES")
    )
    pfseaplantleg = (
        get_sum("PF_SEAFD_HI")
        + get_sum("PF_SEAFD_LOW")
        + get_sum("PF_NUTSDS")
        + get_sum("PF_SOY")
        + get_sum("PF_LEGUMES")
    )

    g_whole = get_sum("G_WHOLE")
    d_total = get_sum("D_TOTAL")
    sodium = get_sum("Sodium")
    g_refined = get_sum("G_REFINED")
    add_sugars = get_sum("ADD_SUGARS")

    result = hei_2020(
        kcal,
        vtotalleg,
        vdrkgrleg,
        f_total,
        fwholefrt,
        g_whole,
        d_total,
        pfallprotleg,
        pfseaplantleg,
        monopoly,
        satfat,
        sodium,
        g_refined,
        add_sugars,
    )

    return result


def extract_hei_components(score_tuple):
    # score_tuple order: C1..C13, TOTAL (TOTAL missing if kcal == 0)
    total = score_tuple[13] if len(score_tuple) > 13 else sum(score_tuple[:13])
    return {
        "HEI2015C1_TOTALVEG": score_tuple[0] if len(score_tuple) > 0 else 0,
        "HEI2015C3_TOTALFRUIT": score_tuple[2] if len(score_tuple) > 2 else 0,
        "HEI2015C5_WHOLEGRAIN": score_tuple[4] if len(score_tuple) > 4 else 0,
        "HEI2015C6_TOTALDAIRY": score_tuple[5] if len(score_tuple) > 5 else 0,
        "HEI2015C7_TOTPROT": score_tuple[6] if len(score_tuple) > 6 else 0,
        "HEI2015C9_FATTYACID": score_tuple[8] if len(score_tuple) > 8 else 0,
        "HEI2015C10_SODIUM": score_tuple[9] if len(score_tuple) > 9 else 0,
        "HEI2015C11_REFINEDGRAIN": score_tuple[10] if len(score_tuple) > 10 else 0,
        "HEI2015C12_SFAT": score_tuple[11] if len(score_tuple) > 11 else 0,
        "HEI2015C13_ADDSUG": score_tuple[12] if len(score_tuple) > 12 else 0,
        "HEI2015_TOTAL_SCORE": total,
    }


# Helper function to standardize size units
def get_standardized_size(row):
    val = row.get("Product Size Values", np.nan)
    unit = row.get("Product Size Units", "")

    if pd.isna(val) or pd.isna(unit):
        return np.nan, None

    unit = str(unit).lower().strip()

    if unit in ["lbs", "lb", "pound", "pounds"]:
        return val * 16.0, "continuous"
    elif unit in ["oz", "ounce", "ounces"]:
        return val, "continuous"
    elif unit in ["kg", "kilogram"]:
        return val * 35.274, "continuous"
    elif unit in ["g", "gram", "grams"]:
        return val * 0.035274, "continuous"

    elif unit in ["fl oz", "fl. oz", "fluid ounce"]:
        return val, "continuous"
    elif unit in ["gal", "gallon"]:
        return val * 128.0, "continuous"
    elif unit in ["qt", "quart"]:
        return val * 32.0, "continuous"
    elif unit in ["pt", "pint"]:
        return val * 16.0, "continuous"
    elif unit in ["l", "liter", "liters"]:
        return val * 33.814, "continuous"
    elif unit in ["ml", "milliliter"]:
        return val * 0.033814, "continuous"

    else:
        return val, "discrete"


def get_recipe_and_ingredients_from_gpt(
    current_items_list, valid_categories, api_key, cuisine_preference=""
):
    """
    valid_categories: unique values in Level_2_Category
    """
    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are a registered-dietitian-style assistant that educates and informs; "
        "you do not provide medical advice or diagnose conditions. Use clear, supportive "
        "language at a 5th-grade reading level. Avoid the terms 'intake' or 'food intake'; "
        "refer to 'purchases' or 'basket'. Optimize for improving the Healthy Eating Index (HEI) "
        "score based on the user’s current purchased items. Favor changes that raise positive "
        "components (vegetables, fruits, whole grains, dairy, protein variety) and reduce moderation "
        "components (refined grains, sodium, added sugars, saturated fats). Output must strictly follow "
        "the provided JSON schema. Do not include extra fields. Use only the given enum values for categories. "
        "When suggesting ingredients, prefer widely available grocery items; keep names concise (product + brand, no sizes)."
    )

    cuisine_line = ""
    cuisine_guidance = ""
    if cuisine_preference:
        cuisine_line = f"- Preferred cuisine: {cuisine_preference}\n"
        cuisine_guidance = (
            "- If a preferred cuisine is provided, align the recipe flavor profile, "
            "staples, and seasonings with that cuisine while still optimizing HEI.\n"
        )

    user_msg = (
        "Context\n"
        f"- Current purchased items (cart): {current_items_list}\n"
        f"- Valid categories (Level_2_Category enum): {valid_categories}\n\n"
        f"{cuisine_line}\n"
        "Objective\n"
        "- Propose one healthy, realistic recipe that best improves the overall HEI profile given what is already in the cart, "
        "and list up to 4 missing ingredients to add.\n\n"
        "Hard Requirements\n"
        "- Select a single recipe that uses several items from the current cart when possible, improves HEI components, and avoids exotic-only items.\n"
        "- For each missing ingredient: choose the category strictly from the provided enum; never invent new categories.\n"
        "- Provide concise cooking instructions: 3–6 short steps, everyday techniques.\n\n"
        f"{cuisine_guidance}"
        "Decision Policy (apply silently)\n"
        "- If vegetables/greens-and-beans are low: prefer dark green vegetables or legumes.\n"
        "- If whole grains are low: use whole grains for the base.\n"
        "- If seafood/plant proteins are low: emphasize beans, tofu, lentils, nuts/seeds, or fish.\n"
        "- If sodium likely high: avoid salty condiments/processed meats; prefer herbs/acid.\n"
        "- If added sugars likely high: avoid sugary sauces; prefer herbs/spices/citrus.\n"
        "- If saturated fat likely high: choose olive/canola oil; favor lean/plant proteins.\n\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recipe_recommendation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "recipe_name": {
                                "type": "string",
                                "description": "The name of the suggested recipe.",
                            },
                            "instructions": {
                                "type": "string",
                                "description": "Brief cooking instructions.",
                            },
                            "missing_ingredients": {
                                "type": "array",
                                "description": "List of up to 4 ingredients missing from the cart.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the ingredient",
                                        },
                                        "category": {
                                            "type": "string",
                                            "description": "The exact category from the provided list.",
                                            "enum": valid_categories,
                                        },
                                    },
                                    "required": ["name", "category"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "recipe_name",
                            "instructions",
                            "missing_ingredients",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise Exception("failed to get response")

        data = json.loads(content)

        return data, data.get("missing_ingredients", [])

    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return None, []


def get_recommendation_reasons_from_gpt(recommendations, api_key):
    """
    recommendations: list of dicts with index, original, recommended, category, price info
    returns dict index -> reason
    """
    if not recommendations:
        return {}

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are a registered-dietitian-style assistant. Provide short, specific reasons "
        "why a recommended product is a healthier swap. Avoid medical advice or diagnoses. "
        "Use 1 sentence per item, 18 words max. If the recommended product is the same as "
        "the original, say it's already a good choice for HEI balance."
    )

    user_msg = (
        "Provide a reason for each recommendation.\n"
        "Return JSON that matches the schema.\n\n"
        f"Recommendations: {json.dumps(recommendations)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_reasons",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasons": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["index", "reason"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["reasons"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = response.choices[0].message.content
        if not content:
            raise Exception("failed to get response")

        data = json.loads(content)
        reasons = {}
        for item in data.get("reasons", []):
            idx = item.get("index")
            reason = item.get("reason", "")
            if idx is None:
                continue
            reasons[int(idx)] = reason
        return reasons
    except Exception as e:
        print(f"Error calling OpenAI for reasons: {e}")
        return {}


def find_best_match_in_pool(ingredient_info, food_pool, current_diet):
    # 1. Filter by Category -> 2. Search by Name -> 3. HEI optimization among candidates
    ingredient_name = ingredient_info.get("name", "")
    target_category = ingredient_info.get("category", None)

    # 1. filtering by category
    subset = food_pool
    if target_category:
        cat_matches = food_pool[food_pool["Level_2_Category"] == target_category]
        if cat_matches.empty:
            cat_matches = food_pool[
                food_pool["Level_2_Category"].astype(str).str.lower()
                == target_category.lower()
            ]

        if not cat_matches.empty:
            subset = cat_matches
        else:
            return None

    # 2. Search name among the filtered subset
    matches = subset[
        subset["Product Name"].str.contains(
            ingredient_name, case=False, na=False, regex=False
        )
    ]
    if matches.empty:
        words = ingredient_name.split()
        if len(words) > 1:
            matches = subset[
                subset["Product Name"].str.contains(
                    words[-1], case=False, na=False, regex=False
                )
            ]
    if matches.empty:
        return None

    # 3. HEI optimization
    best_candidate = None
    best_score = -1e9

    candidates_to_check = matches
    if len(matches) > 20:
        candidates_to_check = matches.sample(n=20)

    for idx, row in candidates_to_check.iterrows():
        row_df = row.to_frame().T
        temp_diet = pd.concat([current_diet, row_df], ignore_index=True)
        score_tuple = calculate_hei_scores_wrapper(temp_diet)
        score = score_tuple[-1]

        if score > best_score:
            best_score = score
            best_candidate = row

    return best_candidate


# ==============================================================================
# 2. Simulated Annealing
# ==============================================================================
def simulated_annealing_fixed_amount(
    diet_df,
    food_pool_df,
    target_hiearchy_level_defined="",
    niter=1000,
    temp0=500,
    use_openai=False,
    openai_api_key="",
    cuisine_preference="",
):
    if len(target_hiearchy_level_defined) != 0:
        if target_hiearchy_level_defined == "FOODCODE":
            original_target_category = diet_df["DESCRIPTION"]
        else:
            original_target_category = diet_df[target_hiearchy_level_defined]
    else:

        def get_row_category(row):
            h_col = row["Recommendation_Hierarchy"]
            if h_col == "FOODCODE":
                return row["DESCRIPTION"]
            else:
                return row[h_col]

        original_target_category = diet_df.apply(get_row_category, axis=1).tolist()

    current_diet = diet_df.copy().reset_index(drop=True)
    food_pool = food_pool_df.copy().reset_index(drop=True)

    if "Original_Product_Name" not in current_diet.columns:
        current_diet["Original_Product_Name"] = current_diet["Product Name"]

    # 1. size standardization
    res_pool = food_pool.apply(get_standardized_size, axis=1, result_type="expand")
    food_pool["_std_val"] = res_pool[0]
    food_pool["_std_type"] = res_pool[1]

    res_diet = current_diet.apply(get_standardized_size, axis=1, result_type="expand")
    current_diet["_std_val"] = res_diet[0]
    current_diet["_std_type"] = res_diet[1]

    # Store Original Info for reference in Phase 1
    if "Product Size Values" in current_diet.columns:
        current_diet["Original_Product_Size_Values"] = current_diet[
            "Product Size Values"
        ]
        current_diet["Original_Product_Size_Units"] = current_diet["Product Size Units"]
        current_diet["Original_Price"] = current_diet["Price"]

    if "Amount" in food_pool.columns and "FoodAmt" not in food_pool.columns:
        food_pool["FoodAmt"] = food_pool["Amount"]
    if "Amount" in current_diet.columns and "FoodAmt" not in current_diet.columns:
        current_diet["FoodAmt"] = current_diet["Amount"]

    common_cols = list(set(current_diet.columns) & set(food_pool.columns))
    exclude_cols = [
        "FoodCode",
        "Food Code",
        "Original_Product_Name",
        "Original_Product_Size_Values",
        "Original_Product_Size_Units",
        "Original_Price",
    ]
    target_cols = [c for c in common_cols if c not in exclude_cols]

    current_scores_tuple = calculate_hei_scores_wrapper(current_diet)
    current_score = current_scores_tuple[-1]
    print(f"Initial Score: {current_score:.4f}")

    best_diet = current_diet.copy(deep=True)
    best_score = current_score
    scores_history = [current_score]

    # # Phase 1: Optimize Existing Items (Simulated Annealing)
    k = 1
    while k < niter:
        temp = temp0 * (0.9999**k)
        next_diet = current_diet.copy(deep=True)
        strategy = 1

        if len(next_diet) > 0:
            idx_to_replace = np.random.choice(next_diet.index)

            # if it's new item
            is_new_item = (
                next_diet.at[idx_to_replace, "Original_Product_Name"]
                == "New Recommendation"
            )

            final_candidates = pd.DataFrame()

            if is_new_item:
                # if it's new item, we can consider all items in the food pool without hierarchy/category filtering
                final_candidates = food_pool
            else:
                # if it's an existing item, we apply hierarchy/category filtering based on the target category and hierarchy level
                if len(target_hiearchy_level_defined) != 0:
                    target_hiearchy_level = target_hiearchy_level_defined
                else:
                    target_hiearchy_level = next_diet.at[
                        idx_to_replace, "Recommendation_Hierarchy"
                    ]

                if (
                    not target_hiearchy_level
                    or target_hiearchy_level not in food_pool.columns
                    or target_hiearchy_level not in next_diet.columns
                ):
                    candidates = food_pool
                else:
                    target_category = next_diet.at[
                        idx_to_replace, target_hiearchy_level
                    ]
                    if pd.isna(target_category) or str(target_category).strip() == "":
                        candidates = food_pool
                    else:
                        candidates = food_pool[
                            food_pool[target_hiearchy_level] == target_category
                        ]

                # price filtering
                original_price = next_diet.at[idx_to_replace, "Original_Price"]
                if not pd.isna(original_price) and not candidates.empty:
                    price_limit = original_price * 1.2
                    candidates = candidates[candidates["Price"] <= price_limit]

                target_std_val = next_diet.at[idx_to_replace, "_std_val"]
                target_std_type = next_diet.at[idx_to_replace, "_std_type"]

                # Size filtering except when getting recommendations from Level_1_Category
                if target_hiearchy_level == "Level_1_Category":
                    final_candidates = candidates
                elif (
                    not candidates.empty
                    and not pd.isna(target_std_val)
                    and not pd.isna(target_std_type)
                ):
                    same_type_candidates = candidates[
                        candidates["_std_type"] == target_std_type
                    ].copy()
                    if not same_type_candidates.empty:
                        same_type_candidates["size_diff"] = (
                            same_type_candidates["_std_val"] - target_std_val
                        ).abs()
                        limit = target_std_val * 0.1
                        strict_matches = same_type_candidates[
                            same_type_candidates["size_diff"] <= limit
                        ]
                        if not strict_matches.empty:
                            final_candidates = strict_matches
                        else:
                            min_diff = same_type_candidates["size_diff"].min()
                            margin = max(0.1, min_diff * 0.05)
                            closest_matches = same_type_candidates[
                                same_type_candidates["size_diff"] <= (min_diff + margin)
                            ]
                            final_candidates = closest_matches
                    else:
                        final_candidates = candidates
                else:
                    final_candidates = candidates

            # Prefer a different product than the original when possible
            if not final_candidates.empty and not is_new_item:
                orig_name = (
                    next_diet.at[idx_to_replace, "Original_Product_Name"]
                    if "Original_Product_Name" in next_diet.columns
                    else ""
                )
                if not orig_name and "Product Name" in next_diet.columns:
                    orig_name = next_diet.at[idx_to_replace, "Product Name"]
                orig_id = None
                if "Walmart Item ID" in next_diet.columns:
                    orig_id = next_diet.at[idx_to_replace, "Walmart Item ID"]
                filtered_candidates = final_candidates.copy()
                if orig_name and "Product Name" in filtered_candidates.columns:
                    filtered_candidates = filtered_candidates[
                        filtered_candidates["Product Name"] != orig_name
                    ]
                if (
                    orig_id is not None
                    and "Walmart Item ID" in filtered_candidates.columns
                ):
                    filtered_candidates = filtered_candidates[
                        filtered_candidates["Walmart Item ID"] != orig_id
                    ]
                if not filtered_candidates.empty:
                    final_candidates = filtered_candidates

            if not final_candidates.empty:
                pool_idx = np.random.choice(final_candidates.index)
                for col in target_cols:
                    next_diet.at[idx_to_replace, col] = food_pool.at[pool_idx, col]
                if (
                    "Product Name" not in target_cols
                    and "Product Name" in food_pool.columns
                ):
                    next_diet.at[idx_to_replace, "Product Name"] = food_pool.at[
                        pool_idx, "Product Name"
                    ]
                next_diet.at[idx_to_replace, "FoodAmt"] = 1.0
            else:
                pass

        new_score_val = calculate_hei_scores_wrapper(next_diet)[-1]

        diff = new_score_val - current_score
        accept = False
        if diff > 0:
            accept = True
        else:
            if np.random.rand() < math.exp(diff / temp):
                accept = True

        if accept:
            current_diet = next_diet
            current_score = new_score_val
            if current_score > best_score:
                best_score = current_score
                best_diet = current_diet.copy(deep=True)
        scores_history.append(current_score)
        k += 1

    # Phase 1 Optimization Done. best_diet contains the optimized version of the original cart.
    optimized_base_diet = best_diet.copy(deep=True)

    # --------------------------------------------------------------------------
    # If getting OpenAI Recommendations
    # --------------------------------------------------------------------------
    recipe_info = None
    recipe_data_out = None
    if use_openai and openai_api_key:
        print("Calling OpenAI for recipe recommendations...")

        # Get a unique list of Level_2_Category values from the food pool to use as valid categories for GPT
        valid_categories = food_pool_df["Level_2_Category"].dropna().unique().tolist()
        valid_categories = [str(c) for c in valid_categories]

        # Use items from the OPTIMIZED diet
        current_items = optimized_base_diet["Product Name"].tolist()

        recipe_data, missing_ingredients = get_recipe_and_ingredients_from_gpt(
            current_items,
            valid_categories,
            openai_api_key,
            cuisine_preference=cuisine_preference,
        )

        # Start Phase 3 with the optimized base diet
        if recipe_data:
            recipe_info = f"Recipe: {recipe_data.get('recipe_name')} - {recipe_data.get('instructions')}"
            print(f"GPT Suggested Recipe: {recipe_data.get('recipe_name')}")
            print(f"Missing Ingredients: {missing_ingredients}")
            recipe_data_out = recipe_data

            # Start Phase 3 with the optimized base diet
            current_diet_context = optimized_base_diet.copy()
            new_items_df_list = []
            added_categories = []

            for ingred_info in missing_ingredients:
                # Find best product for this ingredient, optimizing HEI when added to the CURRENT context
                match_row = find_best_match_in_pool(
                    ingred_info, food_pool, current_diet_context
                )

                if match_row is not None:
                    match_df = match_row.to_frame().T

                    ingred_name = ingred_info.get("name", "Ingredient")
                    ingred_cat = ingred_info.get("category", "Recipe Ingredient")
                    match_df["Original_Product_Name"] = f"AI Rec: {ingred_name}"

                    # New items don't have original specs
                    match_df["Original_Price"] = np.nan
                    match_df["Original_Size_Value"] = np.nan
                    match_df["Original_Size_Unit"] = np.nan

                    new_items_df_list.append(match_df)
                    added_categories.append(ingred_cat)

                    # Update context for the next ingredient search
                    current_diet_context = pd.concat(
                        [current_diet_context, match_df], ignore_index=True
                    )

            if new_items_df_list:
                # Combine optimized base + new AI items
                best_diet = current_diet_context
                best_score = calculate_hei_scores_wrapper(best_diet)[-1]

                # Update category tracking lists
                if isinstance(original_target_category, list):
                    original_target_category = (
                        original_target_category + added_categories
                    )
                else:
                    original_target_category = (
                        list(original_target_category) + added_categories
                    )

    final_table = best_diet[
        [
            "Original_Product_Name",
            "Product Name",
            "Original_Product_Size_Values",
            "Original_Product_Size_Units",
            "Original_Price",
            "Product Size Values",
            "Product Size Units",
            "Price",
        ]
    ].copy()

    final_table = final_table._rename(
        columns={
            "Original_Product_Name": "Original_Food",
            "Product Name": "New_Food",
            "Original_Product_Size_Values": "Original_Size_Value",
            "Original_Product_Size_Units": "Original_Size_Unit",
            "Product Size Values": "New_Size_Value",
            "Product Size Units": "New_Size_Unit",
            "Price": "New_Price",
        }
    )

    final_table["Target_Category"] = original_target_category

    if len(target_hiearchy_level_defined) != 0:
        final_table["Target_Hierarchy_Level"] = target_hiearchy_level_defined
    else:
        orig_len = len(diet_df)
        ai_added_len = len(best_diet) - orig_len
        orig_hierarchy = diet_df["Recommendation_Hierarchy"].tolist()
        final_table["Target_Hierarchy_Level"] = (
            orig_hierarchy + ["Level_2_Category"] * ai_added_len
        )

    recommendation_reasons = {}
    if use_openai and openai_api_key:
        rec_payload = []
        for idx, row in final_table.reset_index(drop=True).iterrows():
            rec_payload.append(
                {
                    "index": int(idx),
                    "original": str(row.get("Original_Food", "")).strip(),
                    "recommended": str(row.get("New_Food", "")).strip(),
                    "category": str(row.get("Target_Category", "")).strip(),
                    "original_price": row.get("Original_Price", ""),
                    "recommended_price": row.get("New_Price", ""),
                }
            )
        recommendation_reasons = get_recommendation_reasons_from_gpt(
            rec_payload, openai_api_key
        )
        if recommendation_reasons:
            final_table["Recommendation_Reason"] = [
                recommendation_reasons.get(i, "")
                for i in range(len(final_table))
            ]

    # Fill text fields safely; coerce NaN to friendly strings for API/CSV
    obj_cols = final_table.select_dtypes(include=["object"]).columns
    final_table[obj_cols] = final_table[obj_cols].fillna("")
    # For prices/sizes, keep numeric NaNs for internal math; they will be
    # stringified downstream before JSON/CSV to avoid literal NaN text

    original_components = extract_hei_components(current_scores_tuple)
    recommended_components = extract_hei_components(
        calculate_hei_scores_wrapper(best_diet)
    )

    return {
        "original_score": scores_history[0],
        "recommended_score": best_score,
        "original_components": original_components,
        "recommended_components": recommended_components,
        "iterated_scores": scores_history,
        "best_diet_table": final_table,
        "best_diet_full": best_diet,
        "recipe_info": recipe_info,
        "recipe_data": recipe_data_out,
    }


def run_recommendation(
    diet_path,
    food_pool_path,
    niter=1000,
    use_openai=False,
    openai_api_key="",
    cuisine_preference="",
):
    diet_data = pd.read_csv(diet_path)
    food_pool = pd.read_csv(food_pool_path)
    api_key = openai_api_key or OPENAI_API_KEY

    result = simulated_annealing_fixed_amount(
        diet_data,
        food_pool,
        niter=niter,
        use_openai=use_openai,
        openai_api_key=api_key,
        cuisine_preference=cuisine_preference,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HEI-optimized recommendations."
    )
    parser.add_argument(
        "--diet", required=True, help="Path to diet CSV (original cart)."
    )
    parser.add_argument("--pool", required=True, help="Path to food pool CSV.")
    parser.add_argument(
        "--out", default="recommended_diet.csv", help="Output CSV path."
    )
    parser.add_argument(
        "--niter", type=int, default=1000, help="Simulated annealing iterations."
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Enable OpenAI recipe recommendations.",
    )
    parser.add_argument(
        "--openai-api-key", default=OPENAI_API_KEY, help="OpenAI API key."
    )
    args = parser.parse_args()

    result = run_recommendation(
        args.diet,
        args.pool,
        niter=args.niter,
        use_openai=args.use_openai,
        openai_api_key=args.openai_api_key,
    )

    print(f"Original Score: {result['original_score']:.2f}")
    print(f"Recommended Score: {result['recommended_score']:.2f}")
    print(result["best_diet_table"])

    result["best_diet_table"].to_csv(args.out, index=False)

    for case in zip(
        result["best_diet_table"]["Original_Food"],
        result["best_diet_table"]["New_Food"],
    ):
        print(f"Replace '{case[0]}' with '{case[1]}'")

    if result.get("recipe_info"):
        print(result["recipe_info"])
