import os
import re
import math
from collections import defaultdict, Counter
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np

# -------- Optional light deps for search stemming (safe if missing)
try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None

# -------- If your saved model used a custom transformer, keeping the class
#          name available helps joblib unpickle. (No-op unless your model needs it.)
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:
    BaseEstimator = object
    class TransformerMixin: pass  # noqa

class TextToVectorTransformer(BaseEstimator, TransformerMixin):
    """Only needed if your joblib pipeline references this class. Kept for compatibility."""
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return np.array(X, dtype=object)  # placeholder; real model usually not using this here

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH   = os.path.join(BASE_DIR, "shop.db")
DATA_CSV  = os.path.join(BASE_DIR, "data", "assignment3_II.csv")
MODEL_P   = os.path.join(BASE_DIR, "models", "review_recommender.joblib")

app = Flask(__name__)
app.secret_key = "dev-secret"  # replace in production
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

BRAND = "winter"
app.jinja_env.globals["BRAND"] = BRAND

# -------------------------
# DB Models (SQLite-friendly)
# -------------------------
class Item(db.Model):
    __tablename__ = "items"

    id = db.Column(db.Integer, primary_key=True)
    source_clothing_id = db.Column(db.Integer, unique=True)  # optional link to raw dataset id

    title = db.Column(db.String, nullable=False)
    description = db.Column(db.Text)
    class_name = db.Column(db.String)
    department_name = db.Column(db.String)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

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

    reviewer_age = db.Column(db.Integer)

    title = db.Column(db.String)
    body = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    recommend_label = db.Column(db.Integer, nullable=False)           # 0/1 (final value, can be overridden by user)
    model_suggested = db.Column(db.Integer, nullable=False, default=0)  # 0/1 (what the model predicted)
    positive_feedback_count = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.CheckConstraint("rating BETWEEN 1 AND 5", name="chk_rating_range"),
        db.CheckConstraint("recommend_label IN (0,1)", name="chk_rec_label"),
        db.Index("idx_reviews_item_created", "item_id", "created_at"),
    )

# -------------------------
# Model loader (Milestone I) — expect a scikit-learn Pipeline
# -------------------------
pipeline = None
if os.path.exists(MODEL_P):
    try:
        pipeline = joblib.load(MODEL_P)
        print("[Model] Loaded:", MODEL_P)
        # Warm-up so first prediction doesn't feel slow
        try:
            pipeline.predict(["warm up text"])
        except Exception:
            pass
    except Exception as e:
        print("[Model] Failed to load:", e)
else:
    print("[Model] File not found:", MODEL_P)

class ModelUnavailable(Exception):
    pass

def predict_label_strict(text: str) -> int:
    """Return exactly 0 or 1 from your saved Pipeline; never fake a value."""
    if pipeline is None:
        raise ModelUnavailable("Model not loaded")
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty input")
    y = pipeline.predict([text])[0]
    y = int(y)
    if y not in (0, 1):
        raise ValueError(f"Model returned non-binary label: {y}")
    return y

# -------------------------
# Search index (lightweight, plural-aware)
# -------------------------
INVERTED = defaultdict(set)
TOKENS_PER_ITEM = {}  # item_id -> Counter(tokens)
WEIGHTS = {"title": 3.0, "class": 2.0, "department": 1.5, "description": 1.0}
STEMMER = PorterStemmer() if PorterStemmer else None

def normalize_token(w: str) -> str:
    w = w.lower()
    w = re.sub(r"[^a-z0-9]+", "", w)
    if len(w) <= 3:
        return w
    if STEMMER:
        try:
            return STEMMER.stem(w)
        except Exception:
            pass
    # fallback plural handling
    if w.endswith("sses"): return w[:-2]
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if w.endswith("es") and not w.endswith("ses"): return w[:-2]
    if w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w

def tokenize(text: str):
    if not text: return []
    return [normalize_token(t) for t in re.findall(r"[A-Za-z0-9']+", text)]

def index_item(item: Item):
    fields = {
        "title": item.title or "",
        "description": item.description or "",
        "class": item.class_name or "",
        "department": item.department_name or "",
    }
    c = Counter()
    for fname, content in fields.items():
        toks = tokenize(content)
        w = WEIGHTS.get(fname, 1.0)
        for t in toks:
            if not t: continue
            INVERTED[t].add(item.id)
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
    if not q_tokens: return []
    candidate_ids = set()
    for t in q_tokens:
        candidate_ids |= INVERTED.get(t, set())
    scored = []
    for item_id in candidate_ids:
        c = TOKENS_PER_ITEM.get(item_id, Counter())
        s = sum(c.get(t, 0) for t in q_tokens)
        if s > 0:
            scored.append((item_id, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [i for i, _ in scored]

# -------------------------
# CSV import helpers / bootstrap
# -------------------------
def load_items_from_csv(path: str = DATA_CSV) -> int:
    if not os.path.exists(path):
        print(f"[Import] CSV not found at {path}")
        return 0
    df = pd.read_csv(path)
    added = 0
    for _, r in df.iterrows():
        title = str(r.get("Clothes Title", "")).strip()
        desc  = str(r.get("Clothes Description", "")).strip()
        cls   = str(r.get("Class Name", "")).strip()
        dept  = str(r.get("Department Name", "")).strip()
        srcid = r.get("Clothing ID", None)
        if not title:
            continue
        item = None
        if pd.notna(srcid):
            try:
                item = Item.query.filter_by(source_clothing_id=int(srcid)).first()
            except Exception:
                item = None
        if not item:
            item = Item.query.filter_by(title=title, class_name=cls, department_name=dept).first()
        if not item:
            item = Item(
                source_clothing_id=int(srcid) if pd.notna(srcid) else None,
                title=title,
                description=desc,
                class_name=cls,
                department_name=dept,
            )
            db.session.add(item)
            added += 1
    db.session.commit()
    print(f"[Import] Added {added} new items.")
    return added

def bootstrap_if_needed():
    db.create_all()
    if Item.query.count() == 0:
        print("[Bootstrap] Items empty; importing CSV…")
        load_items_from_csv()
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
        if ids:
            items = Item.query.filter(Item.id.in_(ids)).all()
            id_to_item = {it.id: it for it in items}
            items = [id_to_item[i] for i in ids if i in id_to_item]
    else:
        items = Item.query.limit(24).all()
    return render_template("index.html", items=items, q=q, count=count)

@app.route("/item/<int:item_id>")
def item_detail(item_id):
    item = Item.query.get_or_404(item_id)
    reviews = Review.query.filter_by(item_id=item.id)\
                          .order_by(Review.created_at.desc())\
                          .all()
    return render_template("item.html", item=item, reviews=reviews)

@app.route("/item/<int:item_id>/review/new", methods=["GET", "POST"])
def new_review(item_id):
    item = Item.query.get_or_404(item_id)

    if request.method == "POST":
        title  = request.form.get("title", "").strip()
        body   = request.form.get("body", "").strip()
        rating = int(request.form.get("rating", "5"))

        if not body:
            flash("Review description is required.", "danger")
            return redirect(request.url)

        # Authoritative prediction from the model
        try:
            suggested = predict_label_strict((title + " " + body).strip())  # 0/1
            print(f"[Model] Final submit predicted {suggested} for: {title} {body}")
        except Exception:
            flash("Prediction unavailable right now. Please try again.", "danger")
            return redirect(request.url)

        # User can override the suggestion from the form; if missing, default to suggested
        final_lbl = request.form.get("recommend_label")
        final_lbl = int(final_lbl) if final_lbl in ("0", "1") else suggested

        rv = Review(
            item_id=item.id,
            title=title,
            body=body,
            rating=max(1, min(5, rating)),
            recommend_label=final_lbl,
            model_suggested=suggested,
        )
        db.session.add(rv)
        db.session.commit()
        flash("Review published.", "success")
        return redirect(url_for("review_detail", review_id=rv.id))

    return render_template("review_form.html", item=item, suggested=None)

@app.route("/suggest", methods=["POST"])
def suggest_label():
    """Return plain text '0' or '1' so the browser JS can set the dropdown."""
    title = request.form.get("title", "").strip()
    return Response("1", mimetype="text/plain")
    body  = request.form.get("body", "").strip()
    text  = (title + " " + body).strip()
    try:
        lbl = predict_label_strict(text)  # 0/1 from your model
        print(f"[Model] /suggest => {lbl} for: {text}")
        return Response(str(lbl), mimetype="text/plain")
    except Exception as e:
        print("[/suggest] error:", e)
        return Response("ERR", status=503, mimetype="text/plain")

@app.route("/reviews/<int:review_id>")
def review_detail(review_id):
    rv = Review.query.get_or_404(review_id)
    return render_template("review_detail.html", rv=rv)

@app.route("/admin/reindex")
def admin_reindex():
    build_index()
    flash("Search index rebuilt.", "info")
    return redirect(url_for("index"))

@app.route("/admin/import_csv")
def admin_import_csv():
    """Import/merge CSV without deleting DB. Use ?wipe=1 to clear items & reviews first."""
    if request.args.get("wipe") == "1":
        Review.query.delete()
        Item.query.delete()
        db.session.commit()
        print("[Import] Wiped items + reviews")
    added = load_items_from_csv()
    build_index()
    flash(f"Imported CSV; added {added} items. Index rebuilt.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
