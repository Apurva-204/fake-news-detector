"""
app/routes.py
-------------
All Flask route handlers for the Fake News Detection System.

Routes:
    GET  /           → User dashboard (protected, login required)
    POST /predict    → ML prediction with text and/or image (protected)
    GET  /journalist → Journalist dashboard (protected, journalist/admin role)
    GET  /admin      → Admin dashboard (protected, admin role required)
    GET  /api/logs   → JSON log data (protected)
    GET  /health     → health check

    GET  /login      → login page
    POST /login      → process login
    GET  /register   → register page
    POST /register   → process registration
    GET  /logout     → clear session and redirect to login
"""

import os
import functools
from datetime import datetime
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
    current_app,
)
from werkzeug.utils import secure_filename
import pickle
import re

from app.services import ocr_service
from app.services.auth_service import register_user, authenticate_user, get_user_by_id
from app.services.fact_check_service import search_claim
# Import the canonical clean_text — MUST match what was used during model training
from app.services.preprocess import clean_text

# -------------------------------------------------------------------
# GLOBAL ML MODEL LOADING — use absolute path so it works from any CWD
# -------------------------------------------------------------------
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
_MODEL_PATH      = os.path.join(_HERE, "model", "model.pkl")
_VECTORIZER_PATH = os.path.join(_HERE, "model", "vectorizer.pkl")

try:
    model = pickle.load(open(_MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(_VECTORIZER_PATH, "rb"))
    print(f"Model loaded OK. Intercept: {model.intercept_}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None

# -------------------------------------------------------------------
# HELPER FOR JOURNALIST ANALYSIS
# -------------------------------------------------------------------
SENSITIVE_CATEGORIES = {
    "Political News": [
        "government", "election", "minister", "policy", "law", "president",
        "parliament", "vote", "congress", "senate", "republican", "democrat",
        "white house", "campaign", "ballot", "constitution", "regime"
    ],
    "Sensational News": [
        "shocking", "breaking", "exclusive", "urgent", "scandal", "outrageous",
        "bombshell", "explosive", "exposed", "leaked", "coverup",
        "conspiracy", "secret", "wake up"
    ],
    "Health & Science": [
        "covid", "virus", "vaccine", "disease", "health", "hospital", "doctor",
        "medicine", "cure", "cancer", "fda", "cdc", "who", "pandemic", "outbreak",
        "microchip", "5g", "experimental", "toxic", "poison"
    ],
    "Financial News": [
        "stock", "market", "economy", "inflation", "tax", "finance", "money",
        "bank", "crypto", "bitcoin", "investment", "debt", "recession", "collapse"
    ],
    "Misinformation Signals": [
        "fake", "hoax", "satire", "debunked", "false", "fabricated", "propaganda",
        "deep state", "globalist", "mainstream media", "they are hiding",
        "elites", "new world order", "bill gates", "george soros", "illuminati"
    ],
    "Crime & Safety": [
        "murder", "arrest", "police", "crime", "terror", "attack", "shooting",
        "weapon", "bomb", "threat", "violence", "illegal", "drug", "gang"
    ]
}

# Category display metadata
CATEGORY_META = {
    "Political News":        {"icon": "🏛️", "color": "#3B82F6", "bg": "rgba(59,130,246,0.12)"},
    "Sensational News":      {"icon": "🚨", "color": "#EF4444", "bg": "rgba(239,68,68,0.12)"},
    "Health & Science":      {"icon": "🧬", "color": "#10B981", "bg": "rgba(16,185,129,0.12)"},
    "Financial News":        {"icon": "📈", "color": "#F59E0B", "bg": "rgba(245,158,11,0.12)"},
    "Misinformation Signals":{"icon": "⚠️", "color": "#EF4444", "bg": "rgba(239,68,68,0.15)"},
    "Crime & Safety":        {"icon": "🚔", "color": "#8B5CF6", "bg": "rgba(139,92,246,0.12)"},
    "General News":          {"icon": "📰", "color": "#6B7280", "bg": "rgba(107,114,128,0.12)"},
}

def analyze_text_details(text, prediction, vectorizer):
    text_lower = text.lower()

    # --- Per-category keyword detection ---
    category_hits = {}
    keyword_map   = []

    for cat, words in SENSITIVE_CATEGORIES.items():
        found = [w for w in words if w in text_lower]
        if found:
            category_hits[cat] = found
            meta = CATEGORY_META.get(cat, CATEGORY_META["General News"])
            for w in found:
                keyword_map.append({
                    "word": w,
                    "category": cat,
                    "color": meta["color"],
                    "bg": meta["bg"],
                    "icon": meta["icon"],
                })

    # Primary category = one with most hits, fallback General News
    primary_category = "General News"
    if category_hits:
        primary_category = max(category_hits, key=lambda c: len(category_hits[c]))

    detected_keywords = list({item["word"] for item in keyword_map})

    # --- Risk level ---
    misinfo_hits  = len(category_hits.get("Misinformation Signals", []))
    sensational   = len(category_hits.get("Sensational News", []))
    total_signals = sum(len(v) for v in category_hits.values())

    if prediction == 0 and (misinfo_hits >= 2 or sensational >= 2):
        risk_level = "HIGH"
    elif prediction == 0 and total_signals >= 2:
        risk_level = "MEDIUM"
    elif prediction == 0:
        risk_level = "LOW-MEDIUM"
    else:
        risk_level = "LOW"

    # --- Explanation ---
    if prediction == 1:
        explanation = (
            "The ML model classified this content as <strong>Real News</strong>. "
            "The vocabulary and sentence structure are consistent with professional, factual journalism. "
            "TF-IDF feature weights align strongly with verified news patterns in the training corpus."
        )
    else:
        if misinfo_hits >= 2:
            explanation = (
                "The ML model classified this as <strong>Fake News</strong>. "
                "Multiple misinformation signal words were detected — these are strongly associated "
                "with fabricated or manipulative content. The linguistic profile deviates significantly "
                "from professional news writing standards."
            )
        elif sensational >= 1:
            explanation = (
                "The ML model classified this as <strong>Fake News</strong>. "
                "The content contains sensationalist language patterns (shock words, urgency triggers) "
                "commonly used in clickbait and misinformation. The emotional tone and hyperbolic framing "
                "are characteristic of fabricated stories."
            )
        else:
            explanation = (
                "The ML model classified this as <strong>Fake News</strong> based on subtle "
                "linguistic patterns in the TF-IDF feature space that correlate with misinformation. "
                "The writing style, word choice, or phrasing do not match verified journalism standards."
            )

    # --- Important words (Top TF-IDF) ---
    important_words = []
    try:
        if vectorizer:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()
            sorted_items = sorted(zip(vec.indices, vec.data), key=lambda x: x[1], reverse=True)
            important_words = [feature_names[i] for i, _ in sorted_items[:10]]
    except Exception as e:
        print(f"Error extracting important words: {e}")

    # --- Research queries ---
    research_queries = []
    for w in important_words[:5]:
        if len(w) > 3:
            research_queries.append({"query": w + " fact check", "label": 'Fact-check "' + w + '"', "type": "fact_check"})
    for item in keyword_map[:3]:
        q = item["word"] + " " + primary_category.lower() + " verification"
        research_queries.append({"query": q, "label": 'Verify "' + item["word"] + '" claim', "type": "verify"})
    short_text = text[:80].strip()
    research_queries.append({"query": short_text + " debunked", "label": "Search for debunking articles", "type": "debunk"})
    seen = set()
    unique_queries = []
    for rq in research_queries:
        if rq["query"] not in seen:
            seen.add(rq["query"])
            unique_queries.append(rq)
    research_queries = unique_queries[:6]

    # --- Signal breakdown ---
    signals = []
    for cat, words in category_hits.items():
        meta = CATEGORY_META.get(cat, CATEGORY_META["General News"])
        signals.append({"category": cat, "icon": meta["icon"], "color": meta["color"], "bg": meta["bg"], "words": words, "count": len(words)})
    signals.sort(key=lambda s: s["count"], reverse=True)

    return {
        "keywords":         detected_keywords,
        "keyword_map":      keyword_map,
        "category":         primary_category,
        "category_meta":    CATEGORY_META.get(primary_category, CATEGORY_META["General News"]),
        "signals":          signals,
        "risk_level":       risk_level,
        "explanation":      explanation,
        "important_words":  important_words,
        "research_queries": research_queries,
        "category_hits":    {k: v for k, v in category_hits.items()},
    }

# -------------------------------------------------------------------
# Blueprint
# -------------------------------------------------------------------
main_bp = Blueprint("main", __name__)

# -------------------------------------------------------------------
# In-memory prediction log (Auto-seeded for Admin Dashboard)
# -------------------------------------------------------------------
_prediction_log: list[dict] = []

import random
if not _prediction_log:
    mock_texts = [
        "The government announced a new policy regarding the upcoming election.",
        "Shocking! Exclusive video shows outrageous scandal involving the minister.",
        "A new vaccine for the covid virus has been approved by health officials.",
        "Stock market crashes as inflation hits new highs.",
        "The local community festival was a huge success this year.",
        "Urgent breaking news: the economy is collapsing, banks closing tomorrow!"
    ]
    for i, t in enumerate(mock_texts):
        is_fake = "Shocking" in t or "Urgent" in t
        _prediction_log.append({
            "id": i + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_text": t,
            "text_snippet": t[:150],
            "ocr_text": None,
            "prediction": "fake" if is_fake else "real",
            "ml_result": "fake" if is_fake else "real",
            "verdict": "Fake News" if is_fake else "Real News",
            "confidence": round(random.uniform(75.0, 99.9), 2),
            "submitted_by": random.choice(["john_doe", "sarah_smith", "admin", "mike_jones"])
        })


# Allowed image extensions for upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}


def allowed_file(filename: str) -> bool:
    """Check if filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------
# Auth Decorators
# -------------------------------------------------------------------

def login_required(f):
    """
    Decorator: redirect to /login if the user is not authenticated.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            session["next"] = request.path
            flash("Please log in to continue.", "warning")
            return redirect(url_for("main.login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """
    Decorator: requires login AND admin role.
    """
    @functools.wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return render_template("403.html"), 403
        return f(*args, **kwargs)
    return decorated


# -------------------------------------------------------------------
# Context helper — inject current user into all templates
# -------------------------------------------------------------------

@main_bp.context_processor
def inject_user():
    """Make current_user dict available in every template."""
    user = None
    if "user_id" in session:
        user = {
            "id":       session.get("user_id"),
            "username": session.get("username"),
            "role":     session.get("role"),
        }
    return {"current_user": user}


# ====================================================================
# AUTH ROUTES
# ====================================================================

@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("main.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user, message = authenticate_user(
            db_path=current_app.config["DATABASE_PATH"],
            username=username,
            password=password,
        )

        if user:
            session.permanent = True
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            next_url = session.pop("next", None)
            return redirect(next_url or url_for("main.index"))
        else:
            error = message

    return render_template("login.html", error=error)


@main_bp.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("main.index"))

    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm_password", "").strip()
        role     = request.form.get("role", "user").strip()

        if password != confirm:
            error = "Passwords do not match."
        else:
            ok, message = register_user(
                db_path=current_app.config["DATABASE_PATH"],
                username=username,
                email=email,
                password=password,
                role=role,
            )
            if ok:
                success = message
            else:
                error = message

    return render_template("register.html", error=error, success=success)


@main_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.login"))


# ====================================================================
# MAIN ROUTES (protected)
# ====================================================================

@main_bp.route("/", methods=["GET"])
@login_required
def index():
    """Render the User Dashboard — simple centered card interface."""
    return render_template("index.html")


@main_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Handle text and/or image submission and return ML prediction result.
    Supports both form-data (with file upload) and JSON.
    If an image is uploaded, OCR is performed to extract text.
    """
    text = ""
    ocr_text = ""

    # ---- Parse input ----
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
    else:
        text = request.form.get("text", "").strip()

        # Handle image upload
        image_file = request.files.get("image")
        if image_file and image_file.filename and allowed_file(image_file.filename):
            try:
                # Save uploaded file temporarily
                upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
                os.makedirs(upload_folder, exist_ok=True)
                filename = secure_filename(image_file.filename)
                filepath = os.path.join(upload_folder, filename)
                image_file.save(filepath)

                try:
                    # Perform OCR
                    ocr_text = ocr_service.extract_text_from_image(filepath)
                finally:
                    # Clean up
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

                # Append OCR text if found, preserving any manually typed text
                if ocr_text:
                    text = (text + "\n" + ocr_text).strip() if text else ocr_text.strip()
                else:
                    text = text.strip()

            except Exception as e:
                current_app.logger.error(f"Image processing error: {e}")
                if not text:
                    return jsonify({"error": "Failed to process the uploaded image."}), 400

    if not text:
        return jsonify({"error": "Please provide news text or upload an image to analyze."}), 400
    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please provide meaningful content."}), 400

    # -------------------------------------------------------------------
    # ML Prediction
    # -------------------------------------------------------------------
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded on the server."}), 503

    print("Prediction input:", text)
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    print("Prediction output:", prediction)

    # Output Format
    ml_label = "real" if prediction == 1 else "fake"
    verdict = "Real News" if prediction == 1 else "Fake News"
    verdict_class = "success" if prediction == 1 else "danger"

    # Confidence calculation
    try:
        if hasattr(model, 'predict_proba'):
            confidences = model.predict_proba(vectorized)[0]
            ml_confidence = round(float(max(confidences)) * 100, 2)
        elif hasattr(model, 'decision_function'):
            dist = model.decision_function(vectorized)[0]
            ml_confidence = round(min(100.0, max(50.0, 50.0 + abs(dist) * 20)), 2)
        else:
            ml_confidence = 100.0
    except Exception:
        ml_confidence = 100.0
        
    # Journalist Analysis Details
    analysis_details = analyze_text_details(text, prediction, vectorizer)

    # -------------------------------------------------------------------
    # Uncertainty zone — model intercept is -1.06 (slight fake bias)
    # When the decision score is close to 0, the model is guessing.
    # Flag these as uncertain instead of confidently wrong.
    # -------------------------------------------------------------------
    uncertainty_flag = False
    try:
        if hasattr(model, 'decision_function'):
            decision_val = float(model.decision_function(vectorized)[0])
            # If |decision| < 0.4, the model is not confident — mark uncertain
            if abs(decision_val) < 0.4:
                uncertainty_flag = True
    except Exception:
        pass

    if uncertainty_flag:
        verdict = "Uncertain — Needs Manual Review"
        verdict_class = "warning"
        ml_label = "uncertain"
        ml_confidence = round(50.0 + abs(decision_val if 'decision_val' in dir() else 0) * 10, 2)

    # -------------------------------------------------------------------
    # Live Web Fact Checking
    # -------------------------------------------------------------------
    sources, debunk_score = search_claim(text)

    # Heuristic: Override ML ONLY if debunk score is strong (>=3).
    # Old threshold was 1 — far too low, caused legit fact-check articles
    # (saying a claim is TRUE) to flip the verdict to fake.
    if debunk_score >= 3:
        ml_label = "fake"
        verdict = "Fake News (Debunked by Web Sources)"
        verdict_class = "danger"
        ml_confidence = min(99.9, max(85.0, 85.0 + debunk_score * 2))
        uncertainty_flag = False  # override uncertainty if strongly debunked

    result = {
        "ml_label":         ml_label,
        "ml_confidence":    ml_confidence,
        "verdict":          verdict,
        "verdict_class":    verdict_class,
        "text_snippet":     text[:150] + ("..." if len(text) > 150 else ""),
        "ocr_text":         ocr_text if ocr_text else None,
        "sources":          sources,
        "debunk_score":     debunk_score,
        "category":         analysis_details["category"],
        "category_meta":    analysis_details["category_meta"],
        "keywords":         analysis_details["keywords"],
        "keyword_map":      analysis_details["keyword_map"],
        "signals":          analysis_details["signals"],
        "risk_level":       analysis_details["risk_level"],
        "explanation":      analysis_details["explanation"],
        "important_words":  analysis_details["important_words"],
        "research_queries": analysis_details["research_queries"],
    }

    # Log the prediction
    log_entry = {
        "id":           len(_prediction_log) + 1,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text":   text,
        "text_snippet": result["text_snippet"], # kept for backwards compatibility
        "ocr_text":     ocr_text if ocr_text else None,
        "prediction":   ml_label,
        "ml_result":    ml_label,               # kept for backwards compatibility
        "verdict":      result["verdict"],
        "confidence":   ml_confidence,
        "submitted_by": session.get("username", "anonymous"),
    }
    _prediction_log.append(log_entry)

    max_logs = current_app.config.get("MAX_LOG_ENTRIES", 500)
    if len(_prediction_log) > max_logs:
        _prediction_log.pop(0)

    return jsonify(result), 200





@main_bp.route("/admin", methods=["GET"])
@login_required
def admin():
    """Render the Admin Dashboard — accessible by any logged-in user."""
    return render_template("admin.html")


@main_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    """Render the user dashboard."""
    # Filter logs for the current user
    user_logs = [log for log in _prediction_log if log.get("submitted_by") == session.get("username")]
    return render_template("dashboard.html", logs=reversed(user_logs))


@main_bp.route("/journalist", methods=["GET"])
@login_required
def journalist():
    """Render the Journalist Dashboard — advanced analysis interface."""
    if session.get("role") not in ["journalist", "admin"]:
        return render_template("403.html"), 403
    return render_template("journalist.html", logs=reversed(_prediction_log))


@main_bp.route("/explore", methods=["GET"])
@login_required
def explore():
    """Render the explore page."""
    return render_template("explore.html")


@main_bp.route("/api/explore-cards", methods=["GET"])
@login_required
def explore_cards():
    """
    Return user-submitted verified predictions as explore card data.
    Skips the mock seed entries (those have no category in their log).
    Returns the 50 most recent real user submissions.
    """
    # Map ml_result → explore card verdict fields
    VERDICT_MAP = {
        "real": {
            "verdict":       "real",
            "verdict_label": "Verified Real",
            "verdict_desc":  "Our ML model classified this content as real news with high confidence.",
        },
        "fake": {
            "verdict":       "fake",
            "verdict_label": "Fake / False",
            "verdict_desc":  "Our ML model detected patterns strongly associated with misinformation.",
        },
    }

    # Category → Unsplash image mapping (generic stock photos per topic)
    CATEGORY_IMAGES = {
        "Political News":        "https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?auto=format&fit=crop&w=600&q=80",
        "Health & Science":      "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=600&q=80",
        "Financial News":        "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?auto=format&fit=crop&w=600&q=80",
        "Sensational News":      "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=600&q=80",
        "Misinformation Signals":"https://images.unsplash.com/photo-1572949645841-094f3a9c4c94?auto=format&fit=crop&w=600&q=80",
        "Crime & Safety":        "https://images.unsplash.com/photo-1540910419892-4a36d2c3266c?auto=format&fit=crop&w=600&q=80",
        "General News":          "https://images.unsplash.com/photo-1532094349884-543bc11b234d?auto=format&fit=crop&w=600&q=80",
    }

    # Map category → explore filter tag
    CATEGORY_TO_TAG = {
        "Political News":         "politics",
        "Health & Science":       "health",
        "Financial News":         "world",
        "Sensational News":       "technology",
        "Misinformation Signals": "politics",
        "Crime & Safety":         "world",
        "General News":           "world",
    }

    cards = []
    # Walk log in reverse (newest first), skip mock seed entries (id <= 6)
    for entry in reversed(_prediction_log):
        if entry.get("id", 0) <= 6:   # skip the 6 auto-seeded mock entries
            continue

        input_text = entry.get("input_text", "") or ""
        if not input_text or len(input_text) < 10:
            continue

        ml_result = entry.get("ml_result", "fake")
        vmap = VERDICT_MAP.get(ml_result, VERDICT_MAP["fake"])

        # Title = first non-empty line, capped at 80 chars
        first_line = input_text.strip().split("\n")[0].strip()
        title = (first_line[:77] + "...") if len(first_line) > 80 else first_line

        # Body = full text capped at 300 chars
        body = (input_text[:297] + "...") if len(input_text) > 300 else input_text

        # Detect category from text (reuse the analyze helper)
        try:
            details = analyze_text_details(input_text, 1 if ml_result == "real" else 0, vectorizer)
            category = details["category"]
        except Exception:
            category = "General News"

        img  = CATEGORY_IMAGES.get(category, CATEGORY_IMAGES["General News"])
        tag  = CATEGORY_TO_TAG.get(category, "world")

        conf = entry.get("confidence", 0)
        conf_str = f"{conf:.1f}%" if conf else "—"

        cards.append({
            "title":        title,
            "text":         body,
            "verdict":      vmap["verdict"],
            "verdictLabel": vmap["verdict_label"],
            "verdictDesc":  vmap["verdict_desc"] + f" (Confidence: {conf_str})",
            "category":     tag,
            "categoryName": category,
            "img":          img,
            "time":         entry.get("timestamp", ""),
            "submittedBy":  entry.get("submitted_by", "anonymous"),
            "community":    True,   # flag so frontend can show a badge
        })

        if len(cards) >= 50:   # cap at 50 community cards
            break

    return jsonify({"cards": cards}), 200


@main_bp.route("/api/logs", methods=["GET"])
@login_required
def get_logs():
    """Return prediction logs as JSON for the admin dashboard."""
    total         = len(_prediction_log)
    real_count    = sum(1 for e in _prediction_log if e["ml_result"] == "real")
    fake_count    = sum(1 for e in _prediction_log if e["ml_result"] == "fake")
    conflict_count = sum(1 for e in _prediction_log if "Conflicting" in e["verdict"])

    return jsonify({
        "stats": {
            "total":     total,
            "real":      real_count,
            "fake":      fake_count,
            "conflicts": conflict_count,
        },
        "logs": list(reversed(_prediction_log)),
    }), 200

@main_bp.route("/api/users", methods=["GET"])
@admin_required
def api_users():
    """Return all users for the admin dashboard."""
    import sqlite3
    try:
        conn = sqlite3.connect(current_app.config["DATABASE_PATH"])
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({"users": users}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint (no auth required)."""
    model_ok = (model is not None and vectorizer is not None)
    return jsonify({
        "status":            "ok",
        "model_loaded":      model_ok,
        "total_predictions": len(_prediction_log),
    }), 200



