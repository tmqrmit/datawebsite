import os, re, math
from collections import defaultdict, Counter
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np
try:
    from nltk.stem import PorterStemmer
except Exception:  # nltk may be missing in minimal environments
    PorterStemmer = None

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, "shop.db")
DATA_CSV = os.path.join(BASE_DIR, "data", "clothing_reviews_m2.csv")
MODEL_P  = os.path.join(BASE_DIR, "models", "review_recommender.joblib")

app = Flask(__name__)
app.secret_key = "dev-secret"  # replace in production
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------
# DB Models (SQLite-friendly)
# -------------------------
class Item(db.Model):
    __tablename__ = "items"

    id = db.Column(db.Integer, primary_key=True)
    # optional: keep a link to the raw dataset id ("Clothing ID")
    source_clothing_id = db.Column(db.Integer, unique=True)

    title = db.Column(db.String, nullable=False)
    description = db.Column(db.Text)
    class_name = db.Column(db.String)
    department_name = db.Column(db.String)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # helpful indexes
    __table_args__ = (
        db.Index("idx_items_title", "title"),
    )

    reviews = db.relationship(
        "Review",
        backref="item",
        lazy=True,
        cascade="all, delete-orphan",
    )


class Review(db.Model):
    __tablename__ = "reviews"

    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(
        db.Integer,
        db.ForeignKey("items.id", ondelete="CASCADE"),
        nullable=False,
    )

    # from dataset (optional to store)
    reviewer_age = db.Column(db.Integer)

    title = db.Column(db.String)
    body = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    recommend_label = db.Column(db.Integer, nullable=False)          # 0/1
    model_suggested = db.Column(db.Integer, nullable=False, default=0)  # 0/1
    positive_feedback_count = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.CheckConstraint("rating BETWEEN 1 AND 5", name="chk_rating_range"),
        db.CheckConstraint("recommend_label IN (0,1)", name="chk_rec_label"),
        db.Index("idx_reviews_item_created", "item_id", "created_at"),
    )

# -------------------------
# Milestone I model loader
# Expectation: a scikit-learn Pipeline saved via joblib.dump(pipeline)
# with either predict_proba or decision_function
# -------------------------
pipeline = None
if os.path.exists(MODEL_P):
    try:
        pipeline = joblib.load(MODEL_P)
        print("[Model] Loaded:", MODEL_P)
    except Exception as e:
        print("[Model] Failed to load:", e)

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_recommend_proba(text: str) -> float:
    """
    Returns probability that review is 'recommended' (label 1).
    Works with:
      - classifiers exposing predict_proba
      - classifiers exposing decision_function (we apply sigmoid)
    """
    if pipeline is None:
        # Fallback: neutral guess
        return 0.5
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([text])[0]
            # assume positive class is at index 1
            return float(proba[1])
        elif hasattr(pipeline, "decision_function"):
            score = pipeline.decision_function([text])[0]
            return float(_sigmoid(score))
        else:
            # predict -> {0,1}; treat as hard 0/1 prob
            pred = pipeline.predict([text])[0]
            return 0.9 if int(pred) == 1 else 0.1
    except Exception as e:
        print("[Model] Predict error:", e)
        return 0.5

# -------------------------
# Lightweight search index
# - normalizes tokens so "dress" ≈ "dresses", "jeans" ≈ "jean"
# - indexes title/description/class/department with weights
# -------------------------
INVERTED = defaultdict(set)
TOKENS_PER_ITEM = {}  # item_id -> Counter(tokens)
WEIGHTS = {"title": 3.0, "class": 2.0, "department": 1.5, "description": 1.0}
STEMMER = PorterStemmer() if PorterStemmer else None

def normalize_token(w: str) -> str:
    """Normalize a token for search indexing.

    Uses a Porter stemmer so that simple morphological variants map to the
    same root form (e.g. ``dress`` and ``dresses`` → ``dress``).  We still
    strip punctuation and short tokens are returned as-is.
    """
    w = w.lower()
    w = re.sub(r"[^a-z0-9]+", "", w)
    if len(w) <= 3:
        return w
    if STEMMER:
        return STEMMER.stem(w)
    # Fallback manual plural handling when NLTK isn't available
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def tokenize(text: str):
    if not text:
        return []
    return [normalize_token(t) for t in re.findall(r"[A-Za-z0-9']+", text)]

def index_item(item: Item):
    fields = {
        "title": item.title,
        "description": item.description or "",
        "class": item.class_name or "",
        "department": item.department_name or "",
    }
    c = Counter()
    for fname, content in fields.items():
        toks = tokenize(content)
        w = WEIGHTS.get(fname, 1.0)
        for t in toks:
            if not t:
                continue
            INVERTED[t].add(item.id)
            # weighted token count
            c[t] += w
    TOKENS_PER_ITEM[item.id] = c

def build_index():
    INVERTED.clear()
    TOKENS_PER_ITEM.clear()
    for item in Item.query.all():
        index_item(item)
    print(f"[Search] Indexed {len(TOKENS_PER_ITEM)} items, {len(INVERTED)} tokens")

def score_items(q: str):
    q_tokens = [t for t in tokenize(q) if t]
    if not q_tokens:
        return []

    # gather candidate item ids
    candidate_ids = set()
    for t in q_tokens:
        candidate_ids |= INVERTED.get(t, set())

    scored = []
    for item_id in candidate_ids:
        c = TOKENS_PER_ITEM[item_id]
        # simple relevance = sum of weighted counts for matched tokens
        s = sum(c.get(t, 0) for t in q_tokens)
        if s > 0:
            scored.append((item_id, s))

    # sort by score desc, then id
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in scored]

# -------------------------
# Bootstrap DB and load data on first run
# -------------------------
def bootstrap_if_needed():
    first_time = not os.path.exists(DB_PATH)
    db.create_all()
    if first_time:
        if not os.path.exists(DATA_CSV):
            print(f"[Bootstrap] Data CSV not found at {DATA_CSV}")
            return
        df = pd.read_csv(DATA_CSV)
        # Expected columns:
        #  'Clothes Title', 'Clothes Description', 'Class Name', 'Department Name'
        # If names differ, adjust here.
        rows = []
        for _, r in df.iterrows():
            title = str(r.get("Clothes Title", "")).strip()
            desc  = str(r.get("Clothes Description", "")).strip()
            cls   = str(r.get("Class Name", "")).strip()
            dept  = str(r.get("Department Name", "")).strip()
            if not title:
                continue
            rows.append(Item(title=title, description=desc, class_name=cls, department_name=dept))
        db.session.add_all(rows)
        db.session.commit()
        print(f"[Bootstrap] Imported {len(rows)} items.")
    build_index()

with app.app_context():
    bootstrap_if_needed()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    items = []
    count = None
    if q:
        ids = score_items(q)
        count = len(ids)
        items = Item.query.filter(Item.id.in_(ids)).all()
        # keep order by score
        id_to_item = {it.id: it for it in items}
        items = [id_to_item[i] for i in ids if i in id_to_item]
    else:
        items = Item.query.limit(24).all()
    return render_template("index.html", items=items, q=q, count=count)

@app.route("/item/<int:item_id>")
def item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    # newest first
    reviews = Review.query.filter_by(item_id=item.id).order_by(Review.created_at.desc()).all()
    return render_template("item.html", item=item, reviews=reviews)

@app.route("/item/<int:item_id>/review/new", methods=["GET", "POST"])
def new_review(item_id):
    item = Item.query.get_or_404(item_id)
    if request.method == "POST":
        title   = request.form.get("title", "").strip()
        body    = request.form.get("body", "").strip()
        rating  = int(request.form.get("rating", "5"))
        # Model suggestion (shown to user, but user can override)
        suggested = int(request.form.get("suggested", "0"))
        final_lbl = int(request.form.get("recommend_label", suggested))

        if not body:
            flash("Review description is required.", "danger")
            return redirect(request.url)

        rv = Review(
            item_id=item.id,
            title=title,
            body=body,
            rating=max(1, min(5, rating)),
            recommend_label=1 if final_lbl == 1 else 0,
            model_suggested=1 if suggested == 1 else 0,
        )
        db.session.add(rv)
        db.session.commit()
        flash("Review published.", "success")
        return redirect(url_for("review_detail", review_id=rv.id))

    # GET → pre-fill suggested label using the model
    # Use both title+body if present later; on GET we show empty form with a hidden suggestion
    # but we’ll compute suggestion live if user entered preview text via JS (optional).
    # For simplicity, we assume body is entered after; you can AJAX it. Here we default to 1.
    return render_template("review_form.html", item=item, suggested=None)

@app.route("/suggest", methods=["POST"])
def suggest_label():
    """
    Small endpoint to compute model suggestion for entered title/body via form (AJAX-friendly).
    Returns '0' or '1'.
    """
    title = request.form.get("title", "").strip()
    body  = request.form.get("body", "").strip()
    text  = (title + " " + body).strip()
    prob = predict_recommend_proba(text) if text else 0.5
    lbl = 1 if prob >= 0.5 else 0
    return str(lbl)

@app.route("/reviews/<int:review_id>")
def review_detail(review_id):
    rv = Review.query.get_or_404(review_id)
    return render_template("review_detail.html", rv=rv)

# Utility: rebuild the index if needed (e.g., after bulk import)
@app.route("/admin/reindex")
def admin_reindex():
    build_index()
    flash("Search index rebuilt.", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
