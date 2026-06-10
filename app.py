"""
AI Skincare Recommendation System — Refactored Backend
=======================================================
Architecture: Hybrid TF-IDF + Concern-Aware Scoring + Category Diversification
"""

import os
import re
import logging
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

possible_dirs = [
    os.path.join(os.path.dirname(__file__), "data"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset")),
]

PRODUCT_CSV = None
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

for pdir in possible_dirs:
    pfile = os.path.join(pdir, "product_info.csv")
    if os.path.exists(pfile):
        PRODUCT_CSV = pfile
        DATA_DIR = pdir
        break

if PRODUCT_CSV is None:
    PRODUCT_CSV = os.path.join(DATA_DIR, "product_info.csv")

REVIEW_FILES = [
    os.path.join(DATA_DIR, "reviews_0-250.csv"),
    os.path.join(DATA_DIR, "reviews_250-500.csv"),
    os.path.join(DATA_DIR, "reviews_500-750.csv"),
    os.path.join(DATA_DIR, "reviews_750-1250.csv"),
    os.path.join(DATA_DIR, "reviews_1250-end.csv"),
]

# ---------------------------------------------------------------------------
# Skincare Filter
# ---------------------------------------------------------------------------

VALID_SKINCARE_SECONDARY = {
    "moisturizer", "cleanser", "serum", "toner", "sunscreen",
    "eye cream", "face mask", "exfoliator", "face oil",
    "mist & essence", "lip treatment", "treatments",
}

SKINCARE_KEYWORD_FALLBACK = [
    "moisturizer", "cleanser", "serum", "toner", "sunscreen",
    "eye cream", "face mask", "exfoliant", "face oil", "essence",
    "spf", "peel", "treatment", "lotion", "emulsion", "ampoule",
    "mist", "balm", "ointment", "primer", "bb cream", "cc cream",
]


def is_skincare_product(row: pd.Series) -> bool:
    primary   = str(row.get("primary_category", "")).lower().strip()
    secondary = str(row.get("secondary_category", "")).lower().strip()
    name      = str(row.get("product_name", "")).lower().strip()
    if primary != "skincare":
        return False
    if secondary in VALID_SKINCARE_SECONDARY:
        return True
    combined = secondary + " " + name
    return any(kw in combined for kw in SKINCARE_KEYWORD_FALLBACK)


# ---------------------------------------------------------------------------
# Slot & Concern Config
# ---------------------------------------------------------------------------

ROUTINE_SLOTS = [
    ("cleanser",     ["cleanser", "face wash", "cleansing", "micellar"],          1),
    ("toner",        ["toner", "essence", "lotion", "facial water"],               1),
    ("serum",        ["serum", "ampoule", "treatment", "booster"],                 2),
    ("moisturizer",  ["moisturizer", "cream", "gel cream", "lotion", "emulsion"], 1),
    ("eye_cream",    ["eye cream", "eye gel", "eye serum"],                        1),
    ("sunscreen",    ["sunscreen", "spf", "sun protection", "uv"],                 1),
    ("mask",         ["mask", "sheet mask", "sleeping mask", "clay mask"],         1),
    ("exfoliant",    ["exfoliant", "scrub", "peel", "aha", "bha", "pha"],          1),
]

SLOT_LABELS = {
    "cleanser":    "Cleanser",
    "toner":       "Toner / Essence",
    "serum":       "Serum / Treatment",
    "moisturizer": "Moisturizer",
    "eye_cream":   "Eye Cream",
    "sunscreen":   "Sunscreen (AM only)",
    "mask":        "Mask / Overnight",
    "exfoliant":   "Exfoliant",
}

CONCERN_INGREDIENT_BOOST: dict[str, list[str]] = {
    "acne":         ["salicylic acid","benzoyl peroxide","niacinamide","zinc","tea tree","sulfur","retinol","azelaic acid","witch hazel","glycolic acid","lactic acid"],
    "large_pores":  ["niacinamide","zinc","retinol","salicylic acid","glycolic acid","clay","kaolin","charcoal"],
    "textured_skin":["aha","bha","glycolic acid","lactic acid","salicylic acid","retinol","vitamin c","alpha arbutin","hyaluronic acid","polyhydroxy acid","pha"],
    "dark_spots":   ["vitamin c","ascorbic acid","niacinamide","tranexamic acid","alpha arbutin","kojic acid","licorice root","azelaic acid","retinol","glycolic acid"],
    "redness":      ["centella asiatica","cica","madecassoside","panthenol","allantoin","aloe vera","green tea","oat","avenanthramide","bisabolol","chamomile","azulene"],
    "dryness":      ["hyaluronic acid","glycerin","ceramide","squalane","shea butter","fatty acid","beta glucan","aloe vera","sodium pca","urea"],
    "anti_aging":   ["retinol","retinal","peptide","niacinamide","vitamin c","collagen","coenzyme q10","adenosine","hyaluronic acid","resveratrol","bakuchiol"],
    "sensitivity":  ["centella asiatica","allantoin","panthenol","ceramide","aloe vera","oat","bisabolol","thermal water","madecassoside"],
}

CONCERN_INGREDIENT_AVOID: dict[str, list[str]] = {
    "acne":       ["coconut oil","isopropyl myristate","lanolin"],
    "redness":    ["alcohol denat","fragrance","parfum","essential oil"],
    "sensitivity":["fragrance","parfum","alcohol denat","essential oil","menthol"],
    "dryness":    ["alcohol denat","sd alcohol"],
}

CONCERN_HIGHLIGHT_BOOST: dict[str, list[str]] = {
    "acne":         ["acne","blemish","pore-minimizing","clarifying","oil-free","non-comedogenic"],
    "large_pores":  ["pore-minimizing","pore-refining","mattifying","oil control"],
    "textured_skin":["exfoliating","resurfacing","smoothing","refining"],
    "dark_spots":   ["brightening","dark spot","hyperpigmentation","uneven tone","radiance"],
    "redness":      ["calming","soothing","redness","sensitive","anti-redness"],
    "dryness":      ["hydrating","moisturizing","plumping","barrier"],
    "anti_aging":   ["anti-aging","firming","lifting","wrinkle","fine lines"],
    "sensitivity":  ["gentle","sensitive","fragrance-free","hypoallergenic","soothing"],
}

SKIN_TYPE_INGREDIENT_BOOST: dict[str, list[str]] = {
    "oily":        ["niacinamide","zinc","salicylic acid","clay","witch hazel"],
    "dry":         ["hyaluronic acid","ceramide","squalane","glycerin","shea butter"],
    "combination": ["niacinamide","hyaluronic acid","zinc"],
    "sensitive":   ["centella asiatica","allantoin","ceramide","panthenol","oat"],
    "normal":      [],
}

SKIN_TYPE_AVOID: dict[str, list[str]] = {
    "oily":     ["coconut oil","mineral oil","lanolin"],
    "dry":      ["alcohol denat","sd alcohol"],
    "sensitive":["fragrance","parfum","alcohol denat","essential oil"],
}

INGREDIENT_BENEFIT: dict[str, str] = {
    "salicylic acid":    "Eksfolian BHA, membersihkan pori",
    "niacinamide":       "Mencerahkan & mengecilkan pori",
    "hyaluronic acid":   "Hidrasi intensif, menarik air ke kulit",
    "retinol":           "Anti-aging, mempercepat regenerasi sel",
    "vitamin c":         "Antioksidan, mencerahkan kulit",
    "ceramide":          "Memperkuat skin barrier",
    "glycerin":          "Humektan, menjaga kelembaban",
    "centella asiatica": "Menenangkan & memperbaiki kulit iritasi",
    "azelaic acid":      "Mengurangi kemerahan & flek hitam",
    "glycolic acid":     "Eksfolian AHA, menghaluskan tekstur",
    "lactic acid":       "Eksfolian AHA, melembabkan sekaligus",
    "benzoyl peroxide":  "Membunuh bakteri penyebab jerawat",
    "zinc":              "Mengontrol sebum & anti-inflamasi",
    "tea tree":          "Antibakteri alami untuk jerawat",
    "alpha arbutin":     "Menghambat melanin, memudarkan flek",
    "kojic acid":        "Mencerahkan bekas jerawat",
    "tranexamic acid":   "Mereduksi hiperpigmentasi",
    "panthenol":         "Melembabkan & memperbaiki skin barrier",
    "allantoin":         "Menenangkan kulit sensitif & iritasi",
    "squalane":          "Melembabkan ringan, cocok semua jenis kulit",
    "shea butter":       "Emolien intensif untuk kulit kering",
    "peptide":           "Merangsang produksi kolagen",
    "adenosine":         "Mengurangi kerutan & anti-aging",
    "kaolin":            "Menyerap sebum berlebih",
    "charcoal":          "Membersihkan pori & kotoran mendalam",
    "aloe vera":         "Menenangkan & menghidrasi kulit",
    "oat":               "Anti-inflamasi untuk kulit sensitif",
    "bisabolol":         "Menenangkan & anti-iritasi",
    "madecassoside":     "Regenerasi kulit & anti-inflamasi",
    "resveratrol":       "Antioksidan kuat, anti-aging",
    "bakuchiol":         "Alternatif retinol alami, lebih lembut",
    "coenzyme q10":      "Antioksidan, melindungi dari radikal bebas",
    "licorice root":     "Mencerahkan & anti-inflamasi",
    "witch hazel":       "Mengecilkan pori, mengontrol minyak",
    "urea":              "Pelembab intensif untuk kulit sangat kering",
    "sodium pca":        "Humektan alami, menjaga hidrasi",
    "beta glucan":       "Memperkuat skin barrier & anti-aging",
    "sulfur":            "Antibakteri untuk jerawat meradang",
    "clay":              "Menyerap minyak & membersihkan pori",
    "ascorbic acid":     "Bentuk vitamin C aktif, antioksidan kuat",
    "collagen":          "Menjaga elastisitas kulit",
    "adapalene":         "Retinoid untuk jerawat & anti-aging",
}

SUNSCREEN_CAP = 1
CATEGORY_CAP  = 2
W_TFIDF       = 0.30
W_INGREDIENT  = 0.35
W_HIGHLIGHT   = 0.10
W_RATING      = 0.15
W_REVIEW_CNT  = 0.10

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

_df: Optional[pd.DataFrame]           = None
_tfidf_matrix                          = None
_vectorizer: Optional[TfidfVectorizer] = None
_review_stats: dict                    = {}


def _load_review_stats() -> dict:
    dfs = []
    for fpath in REVIEW_FILES:
        if os.path.exists(fpath):
            try:
                chunk = pd.read_csv(
                    fpath, low_memory=False,
                    usecols=lambda c: c in {
                        "product_id", "author_id", "rating",
                        "review_text", "skin_type", "is_recommended",
                    }
                )
                dfs.append(chunk)
                logger.info(f"  Loaded reviews: {fpath} ({len(chunk):,} rows)")
            except Exception as e:
                logger.warning(f"  Skip {fpath}: {e}")

    if not dfs:
        logger.warning("No review files found — review stats will be empty.")
        return {}

    reviews = pd.concat(dfs, ignore_index=True)
    reviews.drop_duplicates(subset=["author_id", "product_id"], keep="first", inplace=True)

    reviews["rating"] = pd.to_numeric(reviews.get("rating"), errors="coerce")
    reviews = reviews[reviews["rating"].between(1, 5)].copy()
    reviews["product_id"] = reviews["product_id"].astype(str)

    stats: dict = {}

    for pid, grp in reviews.groupby("product_id"):
        ratings = grp["rating"].dropna()
        dist = {i: int((ratings == i).sum()) for i in range(1, 6)}

        has_text = grp["review_text"].notna() & (grp["review_text"].astype(str).str.strip() != "")
        pool = grp[has_text].copy() if has_text.any() else grp.copy()
        pool = pool.sort_values("rating", ascending=False).head(8)
        samples = []
        for _, row in pool.iterrows():
            if len(samples) >= 3:
                break
            text = str(row.get("review_text", "")).strip()
            if len(text) < 20:
                continue
            if len(text) > 200:
                text = text[:197] + "..."
            samples.append({
                "rating":    int(row["rating"]),
                "text":      text,
                "skin_type": str(row.get("skin_type", "")).strip().title() or "Unknown",
            })

        stats[str(pid)] = {
            "review_count":   int(len(ratings)),
            "avg_rating":     round(float(ratings.mean()), 2),
            "rating_dist":    dist,
            "sample_reviews": samples,
            "recommend_pct":  None,
        }

        if "is_recommended" in grp.columns:
            rec = pd.to_numeric(grp["is_recommended"], errors="coerce").dropna()
            if len(rec) > 0:
                stats[str(pid)]["recommend_pct"] = round(float(rec.mean()) * 100, 1)

    logger.info(f"Review stats built for {len(stats):,} products.")
    return stats


def load_data() -> pd.DataFrame:
    global _df, _tfidf_matrix, _vectorizer, _review_stats
    if _df is not None:
        return _df

    logger.info("Loading product data...")
    df = pd.read_csv(PRODUCT_CSV, low_memory=False)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = ["product_name", "brand_name", "ingredients", "secondary_category",
                 "rating", "highlights", "price_usd", "primary_category", "product_id"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    text_cols = ["product_name", "brand_name", "ingredients", "highlights",
                  "secondary_category", "primary_category"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

    df["product_id"] = df["product_id"].astype(str)

    n_before = len(df)
    df = df[df.apply(is_skincare_product, axis=1)].copy()
    logger.info(f"Skincare filter: {n_before:,} → {len(df):,} ({n_before-len(df):,} removed)")

    df["rating"]    = pd.to_numeric(df.get("rating"),    errors="coerce").fillna(3.5)
    df["reviews"]   = pd.to_numeric(df.get("reviews"),   errors="coerce").fillna(0)
    df["price_usd"] = pd.to_numeric(df.get("price_usd"), errors="coerce").fillna(0)

    df["rating_norm"]  = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-9)
    log_reviews        = np.log1p(df["reviews"])
    df["reviews_norm"] = log_reviews / (log_reviews.max() + 1e-9)

    df["corpus"] = (
        df["product_name"] + " " +
        df["secondary_category"] + " " +
        df["highlights"] + " " +
        df["ingredients"]
    )

    logger.info("Building TF-IDF matrix...")
    _vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                                   stop_words="english", sublinear_tf=True)
    _tfidf_matrix = _vectorizer.fit_transform(df["corpus"])

    _df = df.reset_index(drop=True)
    logger.info(f"Ready: {len(_df):,} skincare products from {_df['brand_name'].nunique():,} brands.")

    logger.info("Loading review stats...")
    _review_stats = _load_review_stats()

    return _df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _ingredient_score(ingredients, boost_terms):
    if not boost_terms or not ingredients: return 0.0
    return min(sum(1 for t in boost_terms if t in ingredients) / max(len(boost_terms), 1), 1.0)

def _penalty_score(ingredients, avoid_terms):
    if not avoid_terms or not ingredients: return 0.0
    return min(sum(1 for t in avoid_terms if t in ingredients) / max(len(avoid_terms), 1), 1.0)

def _highlight_score(highlights, highlight_terms):
    if not highlight_terms or not highlights: return 0.0
    return min(sum(1 for t in highlight_terms if t in highlights) / max(len(highlight_terms), 1), 1.0)


def compute_concern_terms(concerns, skin_type):
    boost, avoid, high = set(), set(), set()
    for c in concerns:
        boost.update(CONCERN_INGREDIENT_BOOST.get(c, []))
        avoid.update(CONCERN_INGREDIENT_AVOID.get(c, []))
        high.update(CONCERN_HIGHLIGHT_BOOST.get(c, []))
    boost.update(SKIN_TYPE_INGREDIENT_BOOST.get(skin_type, []))
    avoid.update(SKIN_TYPE_AVOID.get(skin_type, []))
    return list(boost), list(avoid), list(high)


def compute_need_score(row, boost_ings, avoid_ings, highlight_terms):
    raw = (0.60 * _ingredient_score(row["ingredients"], boost_ings)
           + 0.25 * _highlight_score(row["highlights"], highlight_terms)
           - 0.50 * _penalty_score(row["ingredients"], avoid_ings))
    return float(np.clip(raw, 0.0, 1.0))


def compute_slot_score(row, slot_keywords):
    combined = row["secondary_category"] + " " + row["product_name"]
    return min(sum(1 for kw in slot_keywords if kw in combined) / max(len(slot_keywords), 1), 1.0)


def build_query_vector(concerns, skin_type, brand):
    terms = list(concerns) + [skin_type]
    if brand: terms.append(brand.lower())
    for c in concerns:
        terms.extend(CONCERN_INGREDIENT_BOOST.get(c, [])[:5])
    return _vectorizer.transform([" ".join(terms)])


def score_products(df, concerns, skin_type, brand):
    boost_ings, avoid_ings, highlight_terms = compute_concern_terms(concerns, skin_type)
    query_vec  = build_query_vector(concerns, skin_type, brand)
    tfidf_sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    df = df.copy()
    df["tfidf_score"] = tfidf_sims
    df["need_score"]  = df.apply(lambda r: compute_need_score(r, boost_ings, avoid_ings, highlight_terms), axis=1)

    brand_lower = brand.lower() if brand else ""
    df["brand_match"] = df["brand_name"].apply(lambda b: 0.10 if (brand_lower and brand_lower in b) else 0.0)
    df["total_score"] = (
        W_TFIDF      * df["tfidf_score"] +
        W_INGREDIENT * df["need_score"]  +
        W_RATING     * df["rating_norm"] +
        W_REVIEW_CNT * df["reviews_norm"] +
                       df["brand_match"]
    ).clip(0.0, 1.0)
    return df


# ---------------------------------------------------------------------------
# Slot Filling
# ---------------------------------------------------------------------------

def is_sunscreen(row):
    combined = row["secondary_category"] + " " + row["product_name"]
    return any(kw in combined for kw in ["sunscreen", "spf", "sun protection", "uv"])


def fill_slots(scored_df, concerns, skin_type, exclude_indices=None):
    """Fill routine slots, optionally excluding already-used indices (wished products)."""
    used_indices: set[int] = set(exclude_indices or [])
    sunscreen_count = 0
    result = {}

    for slot_name, slot_keywords, n_take in ROUTINE_SLOTS:
        if slot_name == "sunscreen":
            n_take = SUNSCREEN_CAP

        mask = scored_df.apply(lambda r: compute_slot_score(r, slot_keywords) > 0.0, axis=1)
        candidates = scored_df[mask][~scored_df[mask].index.isin(used_indices)].copy()

        if candidates.empty:
            result[slot_name] = []
            continue

        candidates["slot_score"]  = candidates.apply(lambda r: compute_slot_score(r, slot_keywords), axis=1)
        candidates["final_score"] = 0.70 * candidates["total_score"] + 0.30 * candidates["slot_score"]
        candidates = candidates.sort_values("final_score", ascending=False)

        picked = []
        category_seen = {}
        for idx, row in candidates.iterrows():
            if len(picked) >= n_take: break
            if is_sunscreen(row):
                if sunscreen_count >= SUNSCREEN_CAP: continue
                sunscreen_count += 1
            cat = row["secondary_category"] or "unknown"
            if category_seen.get(cat, 0) >= CATEGORY_CAP: continue
            picked.append(format_product(row, idx, concerns))
            used_indices.add(idx)
            category_seen[cat] = category_seen.get(cat, 0) + 1

        result[slot_name] = picked
    return result


# ---------------------------------------------------------------------------
# ★ WISHED PRODUCT — Check compatibility
# ---------------------------------------------------------------------------

# Batas skor minimum agar produk dianggap "compatible"
WISHED_COMPATIBILITY_THRESHOLD = 0.10

# Batas maksimum bahan "avoid" yang masih ditoleransi
WISHED_AVOID_TOLERANCE = 0.20


def check_wished_product(product_id_str: str, skin_type: str, concerns: list[str]) -> dict:
    """
    Cek apakah produk pilihan user kompatibel dengan profil kulit mereka.

    Returns dict dengan keys:
        found        : bool — apakah produk ditemukan di dataset
        compatible   : bool — apakah produk cocok untuk kulit user
        product      : dict|None — data produk (jika found)
        warnings     : list[str] — daftar peringatan ingredients
        match_score  : float — skor kecocokan 0–1
        avoid_matched: list[str] — bahan yang sebaiknya dihindari user
    """
    df = load_data()

    # Cari produk berdasarkan product_id
    matches = df[df["product_id"] == product_id_str]
    if matches.empty:
        return {"found": False, "compatible": False, "product": None,
                "warnings": [], "match_score": 0.0, "avoid_matched": []}

    row = matches.iloc[0]
    idx = matches.index[0]

    ingredients = row["ingredients"]

    # Hitung boost score
    boost_ings, avoid_ings, highlight_terms = compute_concern_terms(concerns, skin_type)
    need_score = compute_need_score(row, boost_ings, avoid_ings, highlight_terms)

    # Cek bahan yang harus dihindari
    avoid_matched = [ing for ing in avoid_ings if ing in ingredients]

    # Hitung avoid penalty ratio
    avoid_penalty = _penalty_score(ingredients, avoid_ings)

    # Compatible jika:
    # - need_score cukup tinggi ATAU
    # - tidak ada bahan berbahaya yang signifikan
    compatible = (need_score >= WISHED_COMPATIBILITY_THRESHOLD) and (avoid_penalty <= WISHED_AVOID_TOLERANCE)

    # Format produk
    formatted = format_product(row, idx, concerns)
    formatted["_skin_type"] = skin_type

    # Tambah review stats
    pid_str = str(row.get("product_id", ""))
    formatted["review_stats"] = _review_stats.get(pid_str, None)

    # Generate reason lines
    from flask import has_request_context
    reason_lines = generate_scientific_reason_lines(formatted, concerns, skin_type)

    # Buat peringatan spesifik per bahan yang harus dihindari
    warnings = []
    AVOID_EXPLANATION = {
        "fragrance":        "dapat memicu iritasi pada kulit sensitif",
        "parfum":           "dapat memicu iritasi pada kulit sensitif",
        "alcohol denat":    "dapat mengeringkan dan mengiritasi kulit",
        "sd alcohol":       "dapat mengeringkan lapisan kulit",
        "essential oil":    "berpotensi memicu reaksi alergi",
        "coconut oil":      "bersifat comedogenic, dapat menyumbat pori",
        "isopropyl myristate": "bersifat comedogenic, berisiko memicu jerawat",
        "lanolin":          "dapat menyumbat pori pada kulit berminyak",
        "menthol":          "dapat memicu sensasi perih pada kulit sensitif",
        "mineral oil":      "bersifat oklusi berat, kurang ideal untuk kulit berminyak",
    }
    for ing in avoid_matched:
        explanation = AVOID_EXPLANATION.get(ing, "perlu diperhatikan untuk jenis kulitmu")
        warnings.append(f"{ing.title()} — {explanation}")

    return {
        "found":         True,
        "compatible":    compatible,
        "product":       formatted,
        "reason_lines":  reason_lines,
        "warnings":      warnings,
        "match_score":   round(need_score, 3),
        "avoid_matched": avoid_matched,
    }


# ---------------------------------------------------------------------------
# Ingredient annotation
# ---------------------------------------------------------------------------

def annotate_ingredients(ingredients_list: list[str], matched_ings: list[str]) -> list[dict]:
    matched_lower = [m.lower() for m in matched_ings]

    result = []
    for ing in ingredients_list:
        ing_lower = ing.lower()

        is_matched = any(
            kw in ing_lower or ing_lower in kw
            for kw in matched_lower
        )

        benefit = ""
        for key, val in INGREDIENT_BENEFIT.items():
            if key in ing_lower or ing_lower in key:
                benefit = val
                break

        result.append({"text": ing, "is_matched": is_matched, "benefit": benefit})
    return result


# ---------------------------------------------------------------------------
# Reason & Format
# ---------------------------------------------------------------------------

def generate_reason(row, concerns, skin_type):
    ingredients = row["ingredients"]
    parts = []
    concern_labels = {
        "acne":"acne & blemishes","large_pores":"enlarged pores",
        "textured_skin":"uneven texture","dark_spots":"dark spots",
        "redness":"redness & irritation","dryness":"dryness",
        "anti_aging":"signs of aging","sensitivity":"skin sensitivity",
    }
    for concern in concerns:
        matched = [ing for ing in CONCERN_INGREDIENT_BOOST.get(concern, []) if ing in ingredients]
        if matched:
            parts.append(f"contains {', '.join(matched[:3])} to help with {concern_labels.get(concern, concern)}")
    skin_matched = [ing for ing in SKIN_TYPE_INGREDIENT_BOOST.get(skin_type, []) if ing in ingredients][:2]
    if skin_matched and skin_type != "normal":
        parts.append(f"formulated with {', '.join(skin_matched)} for {skin_type} skin")
    if not parts:
        r = row.get("rating", 0)
        parts.append(f"highly rated ({r:.1f}/5)" if r >= 4.5 else "a trusted pick in its category")
    return "; ".join(parts).capitalize() + "."


def get_matched_ingredients(row, concerns):
    ingredients = row["ingredients"]
    matched = set()
    for concern in concerns:
        for kw in CONCERN_INGREDIENT_BOOST.get(concern, []):
            if kw in ingredients:
                matched.add(kw)
    skin_type = row.get("_skin_type", "")
    for kw in SKIN_TYPE_INGREDIENT_BOOST.get(skin_type, []):
        if kw in ingredients:
            matched.add(kw)
    return sorted(matched)[:8]


def format_product(row, idx, concerns):
    skin_type = row.get("_skin_type", "")
    raw_ingredients = row.get("ingredients", "")
    ingredients_list = [ing.strip().title() for ing in raw_ingredients.split(",") if ing.strip()]
    matched_ings = get_matched_ingredients(row, concerns)

    return {
        "product_id":            int(idx),
        "product_name":          row.get("product_name", "").title(),
        "brand_name":            row.get("brand_name", "").title(),
        "secondary_category":    row.get("secondary_category", "").title(),
        "price_usd":             round(float(row.get("price_usd", 0)), 2),
        "rating":                round(float(row.get("rating", 0)), 1),
        "reviews":               int(row.get("reviews", 0)),
        "highlights":            row.get("highlights", ""),
        "total_score":           round(float(row.get("total_score", 0)), 4),
        "matched_ingredients":   matched_ings,
        "ingredients":           ingredients_list,
        "ingredients_raw":       raw_ingredients,
        "reason":                generate_reason(row, concerns, skin_type),
        "product_id_str":        str(row.get("product_id", "")),
        "annotated_ingredients": annotate_ingredients(ingredients_list, matched_ings),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def get_recommendations(skin_type, concerns, brand="", wished_product_id=""):
    df = load_data()
    skin_type = skin_type.lower().strip() if skin_type else "normal"
    concerns  = [c.lower().strip() for c in concerns if c] if concerns else []
    brand     = brand.strip() if brand else ""

    scored_df = score_products(df, concerns, skin_type, brand)
    scored_df = scored_df.copy()
    scored_df["_skin_type"] = skin_type

    # ── Proses produk pilihan user ──────────────────────────────────────────
    wished_result = None
    exclude_indices = []

    if wished_product_id and wished_product_id.strip():
        wished_result = check_wished_product(wished_product_id.strip(), skin_type, concerns)
        if wished_result["found"] and wished_result["compatible"]:
            # Exclude dari slot biasa supaya tidak dobel
            exclude_indices = [wished_result["product"]["product_id"]]

    # ── Fill routine slots (tanpa produk pilihan jika sudah compatible) ─────
    slots = fill_slots(scored_df, concerns, skin_type, exclude_indices=exclude_indices)

    total_recs = sum(len(v) for v in slots.values())

    return {
        "status":        "success",
        "skin_type":     skin_type,
        "concerns":      concerns,
        "brand":         brand,
        "wished_result": wished_result,
        "summary":       {"total_recommendations": total_recs,
                          "slots_filled": sum(1 for v in slots.values() if v)},
        "routine": {
            sn: {"label": SLOT_LABELS.get(sn, sn.replace("_", " ").title()), "products": prods}
            for sn, prods in slots.items()
        },
    }


# ---------------------------------------------------------------------------
# Indonesian Reason Generator
# ---------------------------------------------------------------------------

CONCERN_LABELS_ID = {
    "acne":"jerawat & komedo","large_pores":"pori-pori besar",
    "textured_skin":"tekstur kulit tidak merata","dark_spots":"flek hitam & bekas jerawat",
    "redness":"kemerahan & iritasi","dryness":"kulit kering & dehidrasi",
    "anti_aging":"penuaan dini & kerutan","sensitivity":"kulit sensitif",
}


def generate_scientific_reason_lines(product, concerns, skin_type):
    ingredients_raw = product.get("ingredients_raw", "").lower()
    lines = []

    for concern in concerns:
        matched = [ing for ing in CONCERN_INGREDIENT_BOOST.get(concern, []) if ing in ingredients_raw]
        if matched:
            display = ", ".join([f"<em>{m.title()}</em>" for m in matched[:3]])
            label = CONCERN_LABELS_ID.get(concern, concern.replace("_", " "))
            lines.append(f"Mengandung {display} yang terbukti secara klinis membantu mengatasi <strong>{label}</strong>.")

    skin_boosts = SKIN_TYPE_INGREDIENT_BOOST.get(skin_type.lower(), [])
    skin_matched = [ing for ing in skin_boosts if ing in ingredients_raw]
    if skin_matched and skin_type.lower() != "normal":
        display = ", ".join([f"<em>{m.title()}</em>" for m in skin_matched[:2]])
        lines.append(f"Diformulasikan dengan {display} yang cocok untuk kulit <strong>{skin_type.title()}</strong>.")

    avoid_list = list(set(
        SKIN_TYPE_AVOID.get(skin_type.lower(), []) +
        [a for c in concerns for a in CONCERN_INGREDIENT_AVOID.get(c, [])]
    ))
    matched_avoid = [ing for ing in avoid_list if ing in ingredients_raw]
    if matched_avoid:
        display = ", ".join([f"<em>{m.title()}</em>" for m in matched_avoid[:2]])
        lines.append(f"⚠ <strong>Perhatian:</strong> Mengandung {display} yang berpotensi memicu reaksi sensitivitas.")

    if not lines:
        r = product.get("rating", 0.0)
        lines.append(f"Sangat direkomendasikan dengan penilaian <strong>{r:.1f}/5.0</strong>." if r >= 4.5
                     else "Pilihan produk tepercaya untuk melengkapi rangkaian perawatan harian Anda.")

    return lines


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

CONCERN_OPTIONS = {
    "acne":"Jerawat & Komedo","large_pores":"Pori-Pori Besar",
    "textured_skin":"Tekstur Kulit Tidak Merata","dark_spots":"Flek Hitam & Bekas Jerawat",
    "redness":"Kemerahan & Iritasi","dryness":"Kulit Kering & Dehidrasi",
    "anti_aging":"Penuaan Dini (Kerutan)","sensitivity":"Kulit Sensitif",
}


@app.route("/")
def index():
    try:
        df = load_data()
        brands = sorted([b.title() for b in df["brand_name"].dropna().unique() if b.strip()])
        # Untuk dropdown produk: kirim list (product_id, display_name) — max 2000 produk teratas by rating
        products_for_dropdown = (
            df.sort_values("rating", ascending=False)
            .head(2000)[["product_id", "product_name", "brand_name"]]
            .drop_duplicates(subset=["product_id"])
            .assign(
                display=lambda d: d["brand_name"].str.title() + " — " + d["product_name"].str.title()
            )
            [["product_id", "display"]]
            .to_dict("records")
        )
    except Exception as exc:
        logger.error(f"Error loading brands: {exc}")
        brands = ["The Ordinary","CeraVe","La Roche-Posay","Paula's Choice","Cosrx"]
        products_for_dropdown = []

    return render_template(
        "index.html",
        concerns=CONCERN_OPTIONS,
        brands=brands,
        products_for_dropdown=products_for_dropdown,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        is_json = request.is_json or (request.content_type == "application/json")
        if is_json:
            data             = request.get_json(force=True, silent=True) or {}
            skin_type        = data.get("skin_type", "normal")
            concerns         = data.get("concerns", [])
            brand            = data.get("brand", "")
            wished_product_id = data.get("wished_product_id", "")
        else:
            skin_type        = request.form.get("skin_type", "Normal")
            concerns         = request.form.getlist("concerns")
            brand            = request.form.get("brand", "").strip()
            wished_product_id = request.form.get("wished_product_id", "")

        if not isinstance(concerns, list):
            concerns = [concerns]

        result = get_recommendations(skin_type, concerns, brand, wished_product_id)

        if is_json:
            return jsonify(result), 200

        # ── Format routine products ──────────────────────────────────────────
        recs_flat = []
        for slot_name, slot_data in result["routine"].items():
            slot_label = slot_data["label"]
            for p in slot_data["products"]:
                p_copy = _enrich_product_for_template(p, concerns, skin_type)
                p_copy["slot_label"] = slot_label
                recs_flat.append(p_copy)

        # ── Format wished product ────────────────────────────────────────────
        wished_display = None
        if result.get("wished_result"):
            wr = result["wished_result"]
            if wr["found"]:
                p = wr["product"]
                p_copy = _enrich_product_for_template(p, concerns, skin_type)
                p_copy["reason_lines"]  = wr.get("reason_lines", [])
                p_copy["warnings"]      = wr.get("warnings", [])
                p_copy["match_score"]   = wr.get("match_score", 0)
                p_copy["compatible"]    = wr.get("compatible", False)
                wished_display = p_copy
            else:
                # Produk tidak ditemukan sama sekali di dataset
                wished_display = {"found": False}

        user_concern_labels = [CONCERN_OPTIONS.get(c, c.title()) for c in concerns if c]

        return render_template(
            "result.html",
            skin_type=skin_type.title(),
            brand=brand.title() if brand else "Semua Brand",
            user_concerns=user_concern_labels,
            recommendations=recs_flat,
            wished=wished_display,
        )

    except Exception as exc:
        logger.exception("Error in /recommend")
        if request.is_json or (request.content_type == "application/json"):
            return jsonify({"status": "error", "message": str(exc)}), 500
        return render_template("result.html", recommendations=[], error=str(exc))


def _enrich_product_for_template(p: dict, concerns: list, skin_type: str) -> dict:
    """Tambah field display untuk template result.html."""
    p_copy = p.copy()
    p_copy["category"]      = p.get("secondary_category", "-")

    price = p.get("price_usd", 0)
    p_copy["price"]         = f"${price:.2f}" if price > 0 else "-"

    rating = p.get("rating", 0)
    p_copy["rating_stars"]  = int(round(rating))

    p_copy["reason_lines"]  = generate_scientific_reason_lines(p, concerns, skin_type)
    p_copy["matched_ings"]  = p.get("matched_ingredients", [])

    pid_str = str(p.get("product_id_str", p.get("product_id", "")))
    p_copy["review_stats"]  = _review_stats.get(pid_str, None)

    return p_copy


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    load_data()
    app.run(debug=False, host="0.0.0.0", port=5000)