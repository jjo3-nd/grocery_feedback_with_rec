from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from scipy.spatial.distance import cosine
import pandas as pd
import pickle
import sys
from pathlib import Path

def query_keyword_search(queries, key_cat_map, keyword_embeddings, sbert_model, too_broad_cat):
    """
    Processes a list of queries and returns:
    - matched_categories: The matched category or keyword if the category is too broad.
    - matched_keywords: The exact matched keyword.
    - is_too_broad_flags: Boolean flags indicating if the match was from a too-broad category.
    - third_recommendation: The WWEIA category ('wweia_food_category_description') corresponding to the matched keyword.
    """

    # Keyword → WWEIA category
    key_cat_dict = pd.Series(key_cat_map['Assigned Category'].values, 
                             index=key_cat_map['Keyword']).to_dict()
    openai_keywords = list(key_cat_map['Keyword'])

    query_embeddings = sbert_model.encode(queries, convert_to_numpy=True)

    matched_categories = []
    matched_keywords = []
    is_too_broad_flags = []
    third_recommendation = []

    for i, query in enumerate(queries):
        # Compute cosine similarity
        sbert_scores_keyword = np.array([
            1 - cosine(query_embeddings[i], keyword_embeddings[j]) 
            for j in range(len(openai_keywords))
        ])

        # Get best match
        ranked_indices_keyword = np.argsort(sbert_scores_keyword)[::-1]
        match_keyword = openai_keywords[ranked_indices_keyword[0]]
        matched_keywords.append(match_keyword)

        wweia_cat = key_cat_dict.get(match_keyword, None)
        third_recommendation.append(wweia_cat)  # Always include this

        if wweia_cat in too_broad_cat:
            matched_categories.append(match_keyword)  # fallback to keyword
            is_too_broad_flags.append(True)
        else:
            matched_categories.append(wweia_cat)
            is_too_broad_flags.append(False)

    return matched_categories, matched_keywords 
#, is_too_broad_flags, third_recommendation


sample_query = ['apple', 'orange', 'peppers', 'milk']


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
walmart_files_dir = BASE_DIR / "data"

keyword_embeddings = pickle.load(open(walmart_files_dir / "keyword_embeddings.pkl", "rb"))
# keyword_index = pickle.load(open(walmart_files_dir / "keyword_index.pkl", "rb"))

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Dataset Loaded")

too_broad_cat = ["Processed soy products", "Other red and orange vegetables", "Other dark green vegetables", "Other starchy vegetables",
                 "Other vegetables and combinations", "Tomato-based condiments", "Soy-based condiments", "Protein and nutritional powders",
                 "Seasonings", "Spices", "Baking Products", "Citrus Fruits"]

key_cat_map = pd.read_csv(walmart_files_dir / "keyword_category_mapping_complete.csv")


first, second = query_keyword_search(sample_query, key_cat_map, keyword_embeddings, sbert_model, too_broad_cat)
print("Categories:", first)
print("Keywords:", second)
