"""
AI Skincare Recommendation System — Hybrid Backend (Full Model Integration)
============================================================================
Arsitektur: Ingredient-Aware CF (PyTorch SVD) + CBF (TF-IDF) + Ingredient Match Score
Artefak model dimuat dari folder model/ (hasil training notebook).
Concern-aware layer (boost/avoid/highlight) dipertahankan dari versi sebelumnya
karena notebook tidak memodelkan skin concerns secara eksplisit.
"""

import os
import re
import pickle
import logging
from typing import Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import joblib
import scipy.sparse
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Path Resolution — folder model/ relatif terhadap app.py
# ---------------------------------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_model_candidates = [
    os.path.join(BASE_DIR, "model", "model"),
    os.path.join(BASE_DIR, "model"),
]
MODEL_DIR = next(
    (d for d in _model_candidates
     if os.path.exists(os.path.join(d, "products_skincare.parquet"))),
    os.path.join(BASE_DIR, "model"),
)

# ---------------------------------------------------------------------------
# PyTorch Model Class — identik dengan notebook agar state_dict bisa dimuat
# ---------------------------------------------------------------------------

class IngredientAwareSVD(nn.Module):
    def __init__(self, n_users, n_items, ing_dim, n_factors=32, ing_proj_dim=32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, n_factors)
        self.item_emb  = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.ing_proj  = nn.Sequential(
            nn.Linear(ing_dim, ing_proj_dim),
            nn.BatchNorm1d(ing_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(ing_proj_dim, n_factors),
        )
        self.global_bias = nn.Parameter(torch.tensor([3.5]))
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items, user_ing_vecs):
        user_vec = self.user_emb(users) + self.ing_proj(user_ing_vecs)
        item_vec = self.item_emb(items)
        user_b   = self.user_bias(users).squeeze(-1)
        item_b   = self.item_bias(items).squeeze(-1)
        dot      = (user_vec * item_vec).sum(dim=1)
        return self.global_bias + user_b + item_b + dot


# ---------------------------------------------------------------------------
# Slot & Concern Config (tidak ada di notebook — dipertahankan dari versi lama)
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
}

SUNSCREEN_CAP = 1
CATEGORY_CAP  = 2

# Bobot hybrid dari notebook (α·CF + β·CBF + γ·IngMatch)
ALPHA = 0.5   # CF weight
BETA  = 0.3   # CBF weight
GAMMA = 0.2   # Ingredient match weight

# Bobot layer gabungan hybrid + concern-aware
W_HYBRID     = 0.55
W_CONCERN    = 0.30
W_RATING     = 0.10
W_REVIEW_CNT = 0.05

# Concern-aware layer weights (dalam compute_need_score)
W_ING_NEED  = 0.60
W_HLT_NEED  = 0.25
W_AVOID_PEN = 0.50


# ---------------------------------------------------------------------------
# State global — semua artefak model
# ---------------------------------------------------------------------------

_products_df: Optional[pd.DataFrame]  = None   # products_skincare.parquet
_reviews_cf:  Optional[pd.DataFrame]  = None   # reviews_cf.parquet (author_id, product_id, rating)
_tfidf_matrix                          = None   # scipy sparse
_tfidf_vectorizer                      = None   # TfidfVectorizer
_torch_model:  Optional[IngredientAwareSVD] = None
_user_ing_matrix: Optional[np.ndarray]= None   # shape (n_users, ING_DIM)
_user_ingredient_profile: Optional[dict] = None
_eval_metadata: Optional[dict]         = None
_encoding_maps: Optional[dict]         = None   # user2idx, product2idx, idx2product, ing_vocab, …
_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Muat semua artefak model dari MODEL_DIR.
    Mengembalikan products DataFrame (selalu tersedia).
    Artefak CF bersifat opsional — jika tidak ada, fallback ke CBF+concern only.
    """
    global _products_df, _reviews_cf, _tfidf_matrix, _tfidf_vectorizer
    global _torch_model, _user_ing_matrix, _user_ingredient_profile
    global _eval_metadata, _encoding_maps, _DEVICE, ALPHA, BETA, GAMMA

    if _products_df is not None:
        return _products_df

    logger.info("=" * 60)
    logger.info("Loading model artifacts from: %s", MODEL_DIR)
    logger.info("=" * 60)

    # ── 1. Products parquet ──────────────────────────────────────────────
    products_path = os.path.join(MODEL_DIR, "products_skincare.parquet")
    if not os.path.exists(products_path):
        raise FileNotFoundError(
            f"products_skincare.parquet tidak ditemukan di {MODEL_DIR}. "
            "Pastikan training notebook sudah dijalankan dan folder model/ tersedia."
        )
    _products_df = pd.read_parquet(products_path)
    logger.info("  ✅ products_skincare: %d rows", len(_products_df))

    # Pastikan kolom teks lowercase untuk matching
    for col in ["ingredients", "highlights", "product_name", "brand_name",
                "secondary_category", "secondary_lower"]:
        if col in _products_df.columns:
            _products_df[col] = _products_df[col].fillna("").astype(str).str.lower().str.strip()

    # Alias kolom clean jika ada
    if "ingredients_clean" in _products_df.columns:
        _products_df["ingredients"] = _products_df["ingredients_clean"]
    if "highlights_clean" in _products_df.columns:
        _products_df["highlights"] = _products_df["highlights_clean"]

    # Kolom category untuk slot matching — gunakan secondary_lower jika ada
    if "secondary_lower" in _products_df.columns and "secondary_category" not in _products_df.columns:
        _products_df["secondary_category"] = _products_df["secondary_lower"]
    elif "secondary_lower" in _products_df.columns:
        # Isi secondary_category dari secondary_lower jika kosong
        mask = _products_df["secondary_category"].str.strip() == ""
        _products_df.loc[mask, "secondary_category"] = _products_df.loc[mask, "secondary_lower"]

    _products_df["product_id"]  = _products_df["product_id"].astype(str)
    _products_df["rating"]      = pd.to_numeric(_products_df.get("rating"), errors="coerce").fillna(3.5)
    _products_df["reviews"]     = pd.to_numeric(_products_df.get("reviews"), errors="coerce").fillna(0)
    _products_df["price_usd"]   = pd.to_numeric(_products_df.get("price_usd"), errors="coerce").fillna(0)
    _products_df["rating_norm"] = _normalize_col(_products_df["rating"])
    log_rev = np.log1p(_products_df["reviews"])
    _products_df["reviews_norm"] = log_rev / (log_rev.max() + 1e-9)

    # ── 2. Reviews CF ────────────────────────────────────────────────────
    reviews_path = os.path.join(MODEL_DIR, "reviews_cf.parquet")
    if os.path.exists(reviews_path):
        _reviews_cf = pd.read_parquet(reviews_path)
        _reviews_cf["author_id"]  = _reviews_cf["author_id"].astype(str)
        _reviews_cf["product_id"] = _reviews_cf["product_id"].astype(str)
        logger.info("  ✅ reviews_cf: %d rows", len(_reviews_cf))
    else:
        logger.warning("  ⚠️  reviews_cf.parquet tidak ditemukan — CF dinonaktifkan")

    # ── 3. Encoding maps ─────────────────────────────────────────────────
    enc_path = os.path.join(MODEL_DIR, "encoding_maps.pkl")
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            _encoding_maps = pickle.load(f)
        logger.info("  ✅ encoding_maps: %d users, %d products",
                    len(_encoding_maps.get("user2idx", {})),
                    len(_encoding_maps.get("product2idx", {})))
    else:
        logger.warning("  ⚠️  encoding_maps.pkl tidak ditemukan")

    # ── 4. TF-IDF vectorizer + matrix ────────────────────────────────────
    tfidf_vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    tfidf_mat_path = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
    if os.path.exists(tfidf_vec_path) and os.path.exists(tfidf_mat_path):
        _tfidf_vectorizer = joblib.load(tfidf_vec_path)
        _tfidf_matrix     = scipy.sparse.load_npz(tfidf_mat_path)
        logger.info("  ✅ TF-IDF matrix: %s", _tfidf_matrix.shape)
    else:
        logger.warning("  ⚠️  TF-IDF artefak tidak ditemukan — CBF dinonaktifkan")

    # ── 5. PyTorch SVD model ─────────────────────────────────────────────
    svd_path = os.path.join(MODEL_DIR, "torch_svd_model.pt")
    if os.path.exists(svd_path) and _encoding_maps is not None:
        try:
            ckpt = torch.load(svd_path, map_location="cpu")
            _torch_model = IngredientAwareSVD(
                n_users     = ckpt["n_users"],
                n_items     = ckpt["n_products"],
                ing_dim     = ckpt["ing_dim"],
                n_factors   = ckpt.get("n_factors", 32),
                ing_proj_dim= ckpt.get("ing_proj_dim", 32),
            )
            _torch_model.load_state_dict(ckpt["model_state_dict"])
            _torch_model.eval()
            logger.info("  ✅ PyTorch SVD: n_users=%d, n_items=%d, ing_dim=%d",
                        ckpt["n_users"], ckpt["n_products"], ckpt["ing_dim"])
        except Exception as e:
            logger.warning("  ⚠️  Gagal load torch_svd_model.pt: %s", e)
    else:
        logger.warning("  ⚠️  torch_svd_model.pt tidak ditemukan — CF dinonaktifkan")

    # ── 6. User ingredient matrix ────────────────────────────────────────
    uing_path = os.path.join(MODEL_DIR, "user_ing_matrix.npy")
    if os.path.exists(uing_path):
        _user_ing_matrix = np.load(uing_path)
        logger.info("  ✅ user_ing_matrix: %s", _user_ing_matrix.shape)
    else:
        logger.warning("  ⚠️  user_ing_matrix.npy tidak ditemukan")

    # ── 7. User ingredient profile ───────────────────────────────────────
    uprofile_path = os.path.join(MODEL_DIR, "user_ingredient_profile.pkl")
    if os.path.exists(uprofile_path):
        with open(uprofile_path, "rb") as f:
            _user_ingredient_profile = pickle.load(f)
        logger.info("  ✅ user_ingredient_profile: %d users", len(_user_ingredient_profile))
    else:
        logger.warning("  ⚠️  user_ingredient_profile.pkl tidak ditemukan")

    # ── 8. Eval metadata (alpha/beta/gamma override) ──────────────────────
    meta_path = os.path.join(MODEL_DIR, "eval_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            _eval_metadata = pickle.load(f)
        ALPHA = float(_eval_metadata.get("hybrid_alpha", ALPHA))
        BETA  = float(_eval_metadata.get("hybrid_beta",  BETA))
        GAMMA = float(_eval_metadata.get("hybrid_gamma", GAMMA))
        logger.info("  ✅ eval_metadata: RMSE=%.4f, Prec@10=%.4f, α=%.2f β=%.2f γ=%.2f",
                    _eval_metadata.get("rmse", 0),
                    _eval_metadata.get("precision_10", 0),
                    ALPHA, BETA, GAMMA)

    logger.info("Load selesai. Produk: %d | CF aktif: %s | CBF aktif: %s",
                len(_products_df),
                "YA" if _torch_model is not None else "TIDAK",
                "YA" if _tfidf_matrix is not None else "TIDAK")
    return _products_df


def _normalize_col(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def _normalize_series(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0.0 + 1.0
    return (s - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# CF Scoring
# ---------------------------------------------------------------------------

def compute_cf_scores(user_id: str, batch_size: int = 512) -> Optional[pd.Series]:
    """
    Prediksi rating semua produk di product2idx untuk user_id.
    Mengembalikan Series{product_id: predicted_rating} atau None jika tidak tersedia.
    """
    if (_torch_model is None or _encoding_maps is None or _user_ing_matrix is None):
        return None
    uid = str(user_id)
    user2idx   = _encoding_maps.get("user2idx", {})
    product2idx= _encoding_maps.get("product2idx", {})
    idx2product= _encoding_maps.get("idx2product", {})
    if uid not in user2idx:
        return None

    uidx    = user2idx[uid]
    ing_vec = torch.tensor(_user_ing_matrix[uidx], dtype=torch.float32)
    all_iidx= list(range(len(product2idx)))
    preds   = []

    _torch_model.eval()
    with torch.no_grad():
        for i in range(0, len(all_iidx), batch_size):
            batch = all_iidx[i:i+batch_size]
            items_t = torch.tensor(batch, dtype=torch.long)
            users_t = torch.full((len(batch),), uidx, dtype=torch.long)
            ing_t   = ing_vec.unsqueeze(0).expand(len(batch), -1)
            scores  = _torch_model(users_t, items_t, ing_t).clamp(1, 5).numpy()
            preds.extend(zip(batch, scores))

    pid_score = {str(idx2product[iidx]): float(sc) for iidx, sc in preds}
    return pd.Series(pid_score)


# ---------------------------------------------------------------------------
# CBF Scoring
# ---------------------------------------------------------------------------

def compute_cbf_scores(reference_product_id: str) -> Optional[pd.Series]:
    """
    Cosine similarity TF-IDF terhadap produk referensi.
    Mengembalikan Series{df_index: similarity} atau None.
    """
    if _tfidf_matrix is None or _products_df is None:
        return None
    product_id_to_idx = _encoding_maps.get("product_id_to_idx") if _encoding_maps else None
    if product_id_to_idx is None:
        # Bangun dari products_df
        product_id_to_idx = {
            pid: i for i, pid in enumerate(_products_df["product_id"])
        }
    ref_pid = str(reference_product_id)
    if ref_pid not in product_id_to_idx:
        return None
    ref_idx = product_id_to_idx[ref_pid]
    ref_vec = _tfidf_matrix[ref_idx]
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(ref_vec, _tfidf_matrix).flatten()
    # Kembalikan sebagai Series berindex product_id
    return pd.Series(sims, index=_products_df["product_id"].values)


# ---------------------------------------------------------------------------
# Ingredient Match Score (per produk × user ingredient profile)
# ---------------------------------------------------------------------------

def compute_ingredient_match_scores(user_id: str) -> Optional[pd.Series]:
    """
    Skor kecocokan ingredient profile user terhadap setiap produk.
    Menggunakan ing_vocab dari encoding_maps dan user_ingredient_profile.
    """
    if _user_ingredient_profile is None or _encoding_maps is None or _products_df is None:
        return None
    uid     = str(user_id)
    profile = _user_ingredient_profile.get(uid, {})
    if not profile:
        return None
    ing_vocab  = _encoding_maps.get("ing_vocab", {})
    ING_DIM    = len(ing_vocab)
    if ING_DIM == 0:
        return None

    # Buat vector user dari profile
    user_vec = np.zeros(ING_DIM, dtype=np.float32)
    for ing, cnt in profile.items():
        if ing in ing_vocab:
            user_vec[ing_vocab[ing]] = np.log1p(cnt)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm

    # Buat matrix produk (satu baris per produk)
    def _pid_vec(pid):
        pid_to_ing = _encoding_maps.get("pid_to_ingredients", {})
        ing_list   = pid_to_ing.get(str(pid), [])
        v = np.zeros(ING_DIM, dtype=np.float32)
        for ing in ing_list:
            if ing in ing_vocab:
                v[ing_vocab[ing]] += 1.0
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    scores = {}
    for pid in _products_df["product_id"]:
        pv = _pid_vec(pid)
        scores[pid] = float(np.dot(user_vec, pv))
    return pd.Series(scores)


def _pick_cbf_reference(user_id: str) -> Optional[str]:
    """Pilih produk referensi CBF: produk dengan rating tertinggi dari history user."""
    if _reviews_cf is None:
        return None
    uid = str(user_id)
    hist = _reviews_cf[(_reviews_cf["author_id"] == uid) & (_reviews_cf["rating"] >= 4)]
    if len(hist) > 0:
        return str(hist.sort_values("rating", ascending=False).iloc[0]["product_id"])
    # Fallback ke produk terpopuler
    if _products_df is not None:
        return str(_products_df.nlargest(1, "rating")["product_id"].iloc[0])
    return None


def attach_hybrid_scores(df: pd.DataFrame, user_id: Optional[str]) -> pd.DataFrame:
    """
    Tambahkan kolom cf_score, cbf_score, ing_score, hybrid_score ke df.
    Jika user_id None atau CF tidak tersedia, CF+ing dikosongkan dan bobot dialihkan ke CBF.
    """
    df = df.copy()
    df["product_id"] = df["product_id"].astype(str)

    have_cf  = False
    have_cbf = False
    have_ing = False

    # ── CF ──────────────────────────────────────────────────────────────
    if user_id and _torch_model is not None:
        cf_series = compute_cf_scores(user_id)
        if cf_series is not None:
            df["cf_score_raw"] = df["product_id"].map(cf_series).fillna(0.0)
            df["cf_score"]     = _normalize_series(df["cf_score_raw"])
            have_cf = True

    if not have_cf:
        df["cf_score_raw"] = 0.0
        df["cf_score"]     = 0.0

    # ── CBF ─────────────────────────────────────────────────────────────
    if _tfidf_matrix is not None:
        ref_pid = _pick_cbf_reference(user_id) if user_id else None
        if ref_pid is None and _products_df is not None:
            ref_pid = str(_products_df.nlargest(1, "rating")["product_id"].iloc[0])
        cbf_series = compute_cbf_scores(ref_pid) if ref_pid else None
        if cbf_series is not None:
            df["cbf_score"] = df["product_id"].map(cbf_series).fillna(0.0)
            df["cbf_score"] = _normalize_series(df["cbf_score"])
            have_cbf = True

    if not have_cbf:
        df["cbf_score"] = 0.0

    # ── Ingredient Match ─────────────────────────────────────────────────
    if user_id and _user_ingredient_profile is not None:
        ing_series = compute_ingredient_match_scores(user_id)
        if ing_series is not None:
            df["ing_score"] = df["product_id"].map(ing_series).fillna(0.0)
            df["ing_score"] = _normalize_series(df["ing_score"])
            have_ing = True

    if not have_ing:
        df["ing_score"] = 0.0

    # ── Final hybrid score (α·CF + β·CBF + γ·Ing) ─────────────────────
    if have_cf:
        alpha, beta, gamma = ALPHA, BETA, GAMMA
    else:
        # Tidak ada CF → redistribusi bobot ke CBF + Ing
        alpha = 0.0
        total = BETA + GAMMA
        beta  = BETA  / total if total > 0 else 0.6
        gamma = GAMMA / total if total > 0 else 0.4

    df["hybrid_score"] = (
        alpha * df["cf_score"] +
        beta  * df["cbf_score"] +
        gamma * df["ing_score"]
    ).clip(0.0, 1.0)

    return df


# ---------------------------------------------------------------------------
# Concern-Aware Scoring
# ---------------------------------------------------------------------------

def _ingredient_score(ingredients: str, boost_terms: list) -> float:
    if not boost_terms or not ingredients:
        return 0.0
    return min(sum(1 for t in boost_terms if t in ingredients) / max(len(boost_terms), 1), 1.0)

def _penalty_score(ingredients: str, avoid_terms: list) -> float:
    if not avoid_terms or not ingredients:
        return 0.0
    return min(sum(1 for t in avoid_terms if t in ingredients) / max(len(avoid_terms), 1), 1.0)

def _highlight_score(highlights: str, highlight_terms: list) -> float:
    if not highlight_terms or not highlights:
        return 0.0
    return min(sum(1 for t in highlight_terms if t in highlights) / max(len(highlight_terms), 1), 1.0)


def compute_concern_terms(concerns: list, skin_type: str):
    boost, avoid, high = set(), set(), set()
    for c in concerns:
        boost.update(CONCERN_INGREDIENT_BOOST.get(c, []))
        avoid.update(CONCERN_INGREDIENT_AVOID.get(c, []))
        high.update(CONCERN_HIGHLIGHT_BOOST.get(c, []))
    boost.update(SKIN_TYPE_INGREDIENT_BOOST.get(skin_type, []))
    avoid.update(SKIN_TYPE_AVOID.get(skin_type, []))
    return list(boost), list(avoid), list(high)


def compute_need_score(row, boost_ings: list, avoid_ings: list, highlight_terms: list) -> float:
    raw = (W_ING_NEED  * _ingredient_score(row["ingredients"], boost_ings)
           + W_HLT_NEED  * _highlight_score(row["highlights"], highlight_terms)
           - W_AVOID_PEN * _penalty_score(row["ingredients"], avoid_ings))
    return float(np.clip(raw, 0.0, 1.0))


def score_products(df: pd.DataFrame, concerns: list, skin_type: str,
                   user_id: Optional[str] = None) -> pd.DataFrame:
    """
    Gabungkan hybrid score (CF+CBF+Ing) dengan concern-aware score.
    total_score = W_HYBRID·hybrid + W_CONCERN·concern + W_RATING·rating + W_REVIEW·reviews
    """
    df = df.copy()
    df["_skin_type"] = skin_type

    # Hybrid component
    df = attach_hybrid_scores(df, user_id)

    # Concern-aware component
    boost_ings, avoid_ings, highlight_terms = compute_concern_terms(concerns, skin_type)
    df["need_score"] = df.apply(
        lambda r: compute_need_score(r, boost_ings, avoid_ings, highlight_terms), axis=1
    )

    df["total_score"] = (
        W_HYBRID     * df["hybrid_score"] +
        W_CONCERN    * df["need_score"]   +
        W_RATING     * df["rating_norm"]  +
        W_REVIEW_CNT * df["reviews_norm"]
    ).clip(0.0, 1.0)

    return df


# ---------------------------------------------------------------------------
# Slot Filling
# ---------------------------------------------------------------------------

def compute_slot_score(row, slot_keywords: list) -> float:
    combined = row["secondary_category"] + " " + row["product_name"]
    return min(sum(1 for kw in slot_keywords if kw in combined) / max(len(slot_keywords), 1), 1.0)


def is_sunscreen(row) -> bool:
    combined = (row["secondary_category"] + " " + row["product_name"]).lower()
    return any(kw in combined for kw in ["sunscreen", "spf", "sun protection", "uv"])


def fill_slots(scored_df: pd.DataFrame, concerns: list, skin_type: str,
               exclude_indices=None) -> dict:
    used_indices: set = set(exclude_indices or [])
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
        category_seen: dict = {}
        for idx, row in candidates.iterrows():
            if len(picked) >= n_take:
                break
            if is_sunscreen(row):
                if sunscreen_count >= SUNSCREEN_CAP:
                    continue
                sunscreen_count += 1
            cat = row["secondary_category"] or "unknown"
            if category_seen.get(cat, 0) >= CATEGORY_CAP:
                continue
            picked.append(format_product(row, idx, concerns))
            used_indices.add(idx)
            category_seen[cat] = category_seen.get(cat, 0) + 1

        result[slot_name] = picked
    return result


# ---------------------------------------------------------------------------
# Wished Product Check
# ---------------------------------------------------------------------------

WISHED_COMPATIBILITY_THRESHOLD = 0.10
WISHED_AVOID_TOLERANCE         = 0.20


def check_wished_product(product_id_str: str, skin_type: str, concerns: list,
                         scored_df: Optional[pd.DataFrame] = None) -> dict:
    """
    Cek apakah produk pilihan user kompatibel dengan profil kulit mereka.
    Menerima scored_df (sudah dihitung) untuk menghindari rekomputasi.
    """
    df = _products_df  # sudah dimuat

    matches = df[df["product_id"] == str(product_id_str)]
    if matches.empty:
        return {"found": False, "compatible": False, "product": None,
                "warnings": [], "match_score": 0.0, "avoid_matched": [],
                "hybrid_score": 0.0, "cf_score": 0.0, "cbf_score": 0.0, "ing_score": 0.0}

    row = matches.iloc[0]
    idx = matches.index[0]

    # Ambil hybrid scores dari scored_df jika tersedia
    hybrid_score = cf_score = cbf_score = ing_score = 0.0
    if scored_df is not None and idx in scored_df.index:
        srow = scored_df.loc[idx]
        hybrid_score = float(srow.get("hybrid_score", 0.0))
        cf_score     = float(srow.get("cf_score", 0.0))
        cbf_score    = float(srow.get("cbf_score", 0.0))
        ing_score    = float(srow.get("ing_score", 0.0))
        row = srow  # gunakan row yang sudah ada _skin_type

    ingredients = row["ingredients"]
    boost_ings, avoid_ings, highlight_terms = compute_concern_terms(concerns, skin_type)
    need_score    = compute_need_score(row, boost_ings, avoid_ings, highlight_terms)
    avoid_matched = [ing for ing in avoid_ings if ing in ingredients]
    avoid_penalty = _penalty_score(ingredients, avoid_ings)
    compatible    = (need_score >= WISHED_COMPATIBILITY_THRESHOLD) and (avoid_penalty <= WISHED_AVOID_TOLERANCE)

    formatted = format_product(row, idx, concerns)
    formatted["_skin_type"]   = skin_type
    formatted["hybrid_score"] = round(hybrid_score, 4)
    formatted["cf_score"]     = round(cf_score, 4)
    formatted["cbf_score"]    = round(cbf_score, 4)
    formatted["ing_score"]    = round(ing_score, 4)

    reason_lines = generate_scientific_reason_lines(formatted, concerns, skin_type)

    AVOID_EXPLANATION = {
        "fragrance":           "dapat memicu iritasi pada kulit sensitif",
        "parfum":              "dapat memicu iritasi pada kulit sensitif",
        "alcohol denat":       "dapat mengeringkan dan mengiritasi kulit",
        "sd alcohol":          "dapat mengeringkan lapisan kulit",
        "essential oil":       "berpotensi memicu reaksi alergi",
        "coconut oil":         "bersifat comedogenic, dapat menyumbat pori",
        "isopropyl myristate": "bersifat comedogenic, berisiko memicu jerawat",
        "lanolin":             "dapat menyumbat pori pada kulit berminyak",
        "menthol":             "dapat memicu sensasi perih pada kulit sensitif",
        "mineral oil":         "bersifat oklusi berat, kurang ideal untuk kulit berminyak",
    }
    warnings = [
        f"{ing.title()} — {AVOID_EXPLANATION.get(ing, 'perlu diperhatikan untuk jenis kulitmu')}"
        for ing in avoid_matched
    ]

    return {
        "found":         True,
        "compatible":    compatible,
        "product":       formatted,
        "reason_lines":  reason_lines,
        "warnings":      warnings,
        "match_score":   round(need_score, 3),
        "avoid_matched": avoid_matched,
        "hybrid_score":  round(hybrid_score, 4),
        "cf_score":      round(cf_score, 4),
        "cbf_score":     round(cbf_score, 4),
        "ing_score":     round(ing_score, 4),
    }


# ---------------------------------------------------------------------------
# Ingredient Annotation & Formatting
# ---------------------------------------------------------------------------

def annotate_ingredients(ingredients_list: list, matched_ings: list) -> list:
    matched_lower = [m.lower() for m in matched_ings]
    result = []
    for ing in ingredients_list:
        ing_lower = ing.lower()
        is_matched = any(kw in ing_lower or ing_lower in kw for kw in matched_lower)
        benefit = ""
        for key, val in INGREDIENT_BENEFIT.items():
            if key in ing_lower or ing_lower in key:
                benefit = val
                break
        result.append({"text": ing, "is_matched": is_matched, "benefit": benefit})
    return result


def generate_reason(row, concerns: list, skin_type: str) -> str:
    ingredients = row["ingredients"]
    parts = []
    concern_labels = {
        "acne": "acne & blemishes", "large_pores": "enlarged pores",
        "textured_skin": "uneven texture", "dark_spots": "dark spots",
        "redness": "redness & irritation", "dryness": "dryness",
        "anti_aging": "signs of aging", "sensitivity": "skin sensitivity",
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


def get_matched_ingredients(row, concerns: list) -> list:
    ingredients = row["ingredients"]
    matched = set()
    for concern in concerns:
        for kw in CONCERN_INGREDIENT_BOOST.get(concern, []):
            if kw in ingredients:
                matched.add(kw)
    skin_type = str(row.get("_skin_type", ""))
    for kw in SKIN_TYPE_INGREDIENT_BOOST.get(skin_type, []):
        if kw in ingredients:
            matched.add(kw)
    return sorted(matched)[:8]


def build_review_stats(product_id: str) -> Optional[dict]:
    """
    Bangun statistik ulasan komunitas yang NYATA dari reviews_cf.parquet
    untuk satu product_id. Mengembalikan None jika tidak ada data ulasan
    sama sekali untuk produk ini (atau reviews_cf tidak tersedia), sehingga
    template tidak menampilkan rating/jumlah ulasan palsu.

    Schema reviews_cf yang dipastikan ada: author_id, product_id, rating.
    Kolom lain (is_recommended, review_text, skin_type, dll) bersifat
    opsional — dicek dulu sebelum dipakai, supaya tetap aman walau
    skema parquet berbeda dari yang diasumsikan.
    """
    if _reviews_cf is None:
        return None

    pid = str(product_id)
    rows = _reviews_cf[_reviews_cf["product_id"] == pid]
    if rows.empty:
        return None

    review_count = int(len(rows))
    avg_rating   = float(rows["rating"].mean())

    # Distribusi bintang 1..5
    rating_dist = {star: 0 for star in [1, 2, 3, 4, 5]}
    for star, cnt in rows["rating"].round().astype(int).clip(1, 5).value_counts().items():
        rating_dist[int(star)] = int(cnt)

    # Recommend % — hanya jika kolomnya ada di parquet
    recommend_pct = None
    if "is_recommended" in rows.columns:
        rec_col = rows["is_recommended"].dropna()
        if len(rec_col) > 0:
            recommend_pct = float(rec_col.astype(bool).mean() * 100)

    # Contoh ulasan (maks 3) — pakai kolom teks/skin_type jika tersedia
    text_col = next((c for c in ["review_text", "text", "review"] if c in rows.columns), None)
    sample_reviews = []
    if text_col:
        sample_rows = rows[rows[text_col].notna() & (rows[text_col].astype(str).str.strip() != "")]
        sample_rows = sample_rows.sample(min(3, len(sample_rows)), random_state=42) if len(sample_rows) else sample_rows
        for _, rv in sample_rows.iterrows():
            sample_reviews.append({
                "rating":    int(round(float(rv.get("rating", 0)))),
                "skin_type": str(rv.get("skin_type", "")).title() if "skin_type" in rows.columns else "",
                "text":      str(rv[text_col]).strip(),
            })

    return {
        "avg_rating":     round(avg_rating, 2),
        "review_count":   review_count,
        "recommend_pct":  recommend_pct,
        "rating_dist":    rating_dist,
        "sample_reviews": sample_reviews,
    }


def format_product(row, idx, concerns: list) -> dict:
    skin_type       = str(row.get("_skin_type", ""))
    raw_ingredients = row.get("ingredients", "")
    ingredients_list= [ing.strip().title() for ing in raw_ingredients.split(",") if ing.strip()]
    matched_ings    = get_matched_ingredients(row, concerns)

    product_id_str = str(row.get("product_id", ""))
    review_stats   = build_review_stats(product_id_str)

    # Jangan tampilkan rating/jumlah ulasan palsu dari agregat Sephora —
    # hanya pakai angka yang berasal dari ulasan nyata di reviews_cf.
    rating_value  = round(review_stats["avg_rating"], 1) if review_stats else None
    reviews_value = review_stats["review_count"] if review_stats else 0

    return {
        "product_id":            int(idx),
        "product_name":          str(row.get("product_name", "")).title(),
        "brand_name":            str(row.get("brand_name", "")).title(),
        "secondary_category":    str(row.get("secondary_category", "")).title(),
        "price_usd":             round(float(row.get("price_usd", 0)), 2),
        "rating":                rating_value,
        "reviews":               reviews_value,
        "review_stats":          review_stats,
        "highlights":            str(row.get("highlights", "")),
        "total_score":           round(float(row.get("total_score", 0)), 4),
        "hybrid_score":          round(float(row.get("hybrid_score", 0)), 4),
        "cf_score":              round(float(row.get("cf_score", 0)), 4),
        "cbf_score":             round(float(row.get("cbf_score", 0)), 4),
        "ing_score":             round(float(row.get("ing_score", 0)), 4),
        "matched_ingredients":   matched_ings,
        "ingredients":           ingredients_list,
        "ingredients_raw":       raw_ingredients,
        "reason":                generate_reason(row, concerns, skin_type),
        "product_id_str":        product_id_str,
        "annotated_ingredients": annotate_ingredients(ingredients_list, matched_ings),
    }


# ---------------------------------------------------------------------------
# Indonesian Reason Generator
# ---------------------------------------------------------------------------

CONCERN_LABELS_ID = {
    "acne":         "jerawat & komedo",
    "large_pores":  "pori-pori besar",
    "textured_skin":"tekstur kulit tidak merata",
    "dark_spots":   "flek hitam & bekas jerawat",
    "redness":      "kemerahan & iritasi",
    "dryness":      "kulit kering & dehidrasi",
    "anti_aging":   "penuaan dini & kerutan",
    "sensitivity":  "kulit sensitif",
}


def generate_scientific_reason_lines(product: dict, concerns: list, skin_type: str) -> list:
    ingredients_raw = product.get("ingredients_raw", "").lower()
    lines = []
    for concern in concerns:
        matched = [ing for ing in CONCERN_INGREDIENT_BOOST.get(concern, []) if ing in ingredients_raw]
        if matched:
            display = ", ".join([f"<em>{m.title()}</em>" for m in matched[:3]])
            label   = CONCERN_LABELS_ID.get(concern, concern.replace("_", " "))
            lines.append(f"Mengandung {display} yang terbukti secara klinis membantu mengatasi <strong>{label}</strong>.")
    skin_boosts  = SKIN_TYPE_INGREDIENT_BOOST.get(skin_type.lower(), [])
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
        r = product.get("rating")
        lines.append(
            f"Sangat direkomendasikan dengan penilaian <strong>{r:.1f}/5.0</strong>." if r is not None and r >= 4.5
            else "Pilihan produk tepercaya untuk melengkapi rangkaian perawatan harian Anda."
        )
    return lines


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

CONCERN_OPTIONS = {
    "acne":         "Jerawat & Komedo",
    "large_pores":  "Pori-Pori Besar",
    "textured_skin":"Tekstur Kulit Tidak Merata",
    "dark_spots":   "Flek Hitam & Bekas Jerawat",
    "redness":      "Kemerahan & Iritasi",
    "dryness":      "Kulit Kering & Dehidrasi",
    "anti_aging":   "Penuaan Dini (Kerutan)",
    "sensitivity":  "Kulit Sensitif",
}


def get_recommendations(skin_type: str, concerns: list, user_id: str = "",
                        wished_product_id: str = "") -> dict:
    df = load_data()
    skin_type = skin_type.lower().strip() if skin_type else "normal"
    concerns  = [c.lower().strip() for c in concerns if c] if concerns else []
    user_id   = user_id.strip() if user_id else ""

    scored_df = score_products(df, concerns, skin_type, user_id or None)
    scored_df = scored_df.copy()
    scored_df["_skin_type"] = skin_type

    # ── Proses produk pilihan user ──────────────────────────────────────
    wished_result   = None
    exclude_indices = []

    if wished_product_id and wished_product_id.strip():
        wished_result = check_wished_product(
            wished_product_id.strip(), skin_type, concerns, scored_df=scored_df
        )
        if wished_result["found"] and wished_result["compatible"]:
            exclude_indices = [wished_result["product"]["product_id"]]

    # ── Fill routine slots ──────────────────────────────────────────────
    slots = fill_slots(scored_df, concerns, skin_type, exclude_indices=exclude_indices)
    total_recs = sum(len(v) for v in slots.values())

    # ── Model info untuk transparansi ───────────────────────────────────
    cf_active  = _torch_model is not None
    cbf_active = _tfidf_matrix is not None
    model_info = {
        "cf_active":        cf_active,
        "cbf_active":       cbf_active,
        "user_known":       bool(user_id and _encoding_maps and
                                 user_id in _encoding_maps.get("user2idx", {})),
        "alpha":            ALPHA,
        "beta":             BETA,
        "gamma":            GAMMA,
        "eval_rmse":        _eval_metadata.get("rmse")       if _eval_metadata else None,
        "eval_precision10": _eval_metadata.get("precision_10") if _eval_metadata else None,
        "eval_ndcg10":      _eval_metadata.get("ndcg_10")    if _eval_metadata else None,
    }

    return {
        "status":        "success",
        "skin_type":     skin_type,
        "concerns":      concerns,
        "user_id":       user_id,
        "wished_result": wished_result,
        "model_info":    model_info,
        "summary":       {
            "total_recommendations": total_recs,
            "slots_filled":          sum(1 for v in slots.values() if v),
        },
        "routine": {
            sn: {"label": SLOT_LABELS.get(sn, sn.replace("_", " ").title()), "products": prods}
            for sn, prods in slots.items()
        },
    }


def _enrich_product_for_template(p: dict, concerns: list, skin_type: str) -> dict:
    """Tambah field display untuk template result.html."""
    p_copy = p.copy()
    p_copy["category"]     = p.get("secondary_category", "-")
    price = p.get("price_usd", 0)
    p_copy["price"]        = f"${price:.2f}" if price > 0 else "-"
    p_copy["rating_stars"] = int(round(p.get("rating") or 0))
    p_copy["reason_lines"] = generate_scientific_reason_lines(p, concerns, skin_type)
    p_copy["matched_ings"] = p.get("matched_ingredients", [])
    return p_copy


@app.route("/")
def index():
    try:
        df = load_data()
        brands = sorted([b.title() for b in df["brand_name"].dropna().unique() if b.strip()])
        products_for_dropdown = (
            df.sort_values("rating", ascending=False)
            .head(2000)[["product_id", "product_name", "brand_name"]]
            .drop_duplicates(subset=["product_id"])
            .assign(display=lambda d: d["brand_name"].str.title() + " — " + d["product_name"].str.title())
            [["product_id", "display"]]
            .to_dict("records")
        )
        # Sample user_ids untuk dropdown (max 50 user yang ada di CF)
        sample_users: list[str] = []
        if _encoding_maps:
            sample_users = list(_encoding_maps.get("user2idx", {}).keys())[:50]
    except Exception as exc:
        logger.error("Error loading index: %s", exc)
        brands = ["The Ordinary", "CeraVe", "La Roche-Posay", "Paula's Choice", "Cosrx"]
        products_for_dropdown = []
        sample_users = []

    return render_template(
        "index.html",
        concerns=CONCERN_OPTIONS,
        brands=brands,
        products_for_dropdown=products_for_dropdown,
        sample_users=sample_users,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        is_json = request.is_json or (request.content_type == "application/json")
        if is_json:
            data             = request.get_json(force=True, silent=True) or {}
            skin_type        = data.get("skin_type", "normal")
            concerns         = data.get("concerns", [])
            user_id          = data.get("user_id", "")
            wished_product_id= data.get("wished_product_id", "")
        else:
            skin_type        = request.form.get("skin_type", "Normal")
            concerns         = request.form.getlist("concerns")
            user_id          = request.form.get("user_id", "").strip()
            wished_product_id= request.form.get("wished_product_id", "")

        if not isinstance(concerns, list):
            concerns = [concerns]

        result = get_recommendations(skin_type, concerns, user_id, wished_product_id)

        if is_json:
            return jsonify(result), 200

        # ── Format routine products ──────────────────────────────────────
        recs_flat = []
        for slot_name, slot_data in result["routine"].items():
            slot_label = slot_data["label"]
            for p in slot_data["products"]:
                p_copy = _enrich_product_for_template(p, concerns, skin_type)
                p_copy["slot_label"] = slot_label
                recs_flat.append(p_copy)

        # ── Format wished product ────────────────────────────────────────
        wished_display = None
        if result.get("wished_result"):
            wr = result["wished_result"]
            if wr["found"]:
                p      = wr["product"]
                p_copy = _enrich_product_for_template(p, concerns, skin_type)
                p_copy["reason_lines"]  = wr.get("reason_lines", [])
                p_copy["warnings"]      = wr.get("warnings", [])
                p_copy["match_score"]   = wr.get("match_score", 0)
                p_copy["compatible"]    = wr.get("compatible", False)
                p_copy["hybrid_score"]  = wr.get("hybrid_score", 0)
                p_copy["cf_score"]      = wr.get("cf_score", 0)
                p_copy["cbf_score"]     = wr.get("cbf_score", 0)
                p_copy["ing_score"]     = wr.get("ing_score", 0)
                wished_display = p_copy
            else:
                wished_display = {"found": False}

        user_concern_labels = [CONCERN_OPTIONS.get(c, c.title()) for c in concerns if c]

        return render_template(
            "result.html",
            skin_type=skin_type.title(),
            user_id=user_id or None,
            user_concerns=user_concern_labels,
            recommendations=recs_flat,
            wished=wished_display,
            model_info=result.get("model_info", {}),
        )

    except Exception as exc:
        logger.exception("Error in /recommend")
        if request.is_json or (request.content_type == "application/json"):
            return jsonify({"status": "error", "message": str(exc)}), 500
        return render_template("result.html", recommendations=[], error=str(exc))


@app.route("/health")
def health():
    load_data()
    return jsonify({
        "status":       "ok",
        "cf_active":    _torch_model is not None,
        "cbf_active":   _tfidf_matrix is not None,
        "products":     len(_products_df) if _products_df is not None else 0,
        "alpha":        ALPHA,
        "beta":         BETA,
        "gamma":        GAMMA,
    }), 200


if __name__ == "__main__":
    load_data()
    app.run(debug=False, host="0.0.0.0", port=5000)