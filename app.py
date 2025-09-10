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

from bs4 import BeautifulSoup
import difflib
import html
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramCollocationFinder, QuadgramAssocMeasures
from nltk.probability import *
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter as _Counter  # avoid shadowing
import gensim.downloader as api
from gensim.models import FastText
import joblib as _joblib
import lightgbm as lgb
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import words, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import numpy as _np
import os as _os
import pandas as _pd
import re as _re
from scipy.sparse import hstack, csr_matrix
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import warnings

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
warnings.filterwarnings('ignore')

# --- NLTK downloads (as in your file; consider trimming later) ---
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger_eng')

# -------- Optional stemming for simple search
try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None

# -------- Keep class name for joblib compatibility (if needed)
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:
    BaseEstimator = object
    class TransformerMixin: pass  # noqa

class TextToVectorTransformer(BaseEstimator, TransformerMixin):
    # (kept as-is)
    def __init__(self):
        self.vocab = None
        self.fasttext_model = None
        self.tfidf_vectorizer = None
        self.data_is_preprocessed = False
    def fit(self, X, y=None):
        self.vocab = read_vocab('vocab_both.txt')
        self.fasttext_model = FastText.load('fasttext_model.model')
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', vocabulary=self.vocab, lowercase=True)
        self.tfidf_vectorizer.fit(X)
        self.data_is_preprocessed = True
        return self
    def transform(self, X):
        if self.vocab is None or self.fasttext_model is None or self.tfidf_vectorizer is None:
            raise RuntimeError("This transformer has not been fitted yet. Call .fit() before .transform().")
        data = pd.DataFrame({'New Review': X if self.data_is_preprocessed else self.text_preprocessing(X)})
        weighted_vectors = self.calc_weighted_vectors(data, 'New Review', self.vocab, self.fasttext_model)
        self.data_is_preprocessed = False
        return weighted_vectors
    def tokenize(self, texts, get_vocab=False, print_process=False):
        regex_tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        ENTITY_RE = re.compile(r"&(?:[A-Za-z]+|#[0-9]+|#x[0-9A-Fa-f]+);")
        corpus = []
        for text in texts:
            tokens = []
            unescaped = html.unescape(text)
            soup = BeautifulSoup(unescaped, 'html.parser')
            text = soup.get_text()
            text = ENTITY_RE.sub(' ', text)
            for sent in sent_tokenize(text):
                words = regex_tokenizer.tokenize(sent)
                words = [w.lower() for w in words]
                tokens.extend(words)
            corpus.append(tokens)
        if get_vocab:
            unique_tokens = sorted({t for doc in corpus for t in doc})
            vocab = {token: idx for idx, token in enumerate(unique_tokens)}
            return corpus, vocab
        return corpus
    def lemmatize(self, corpus, print_process=False):
        result_corpus = []
        pos_map = {'ADJ':'a','ADP':'s','ADV':'r','NOUN':'n','VERB':'v'}
        lemmatizer = WordNetLemmatizer()
        for doc in corpus:
            doc_with_tag = nltk.pos_tag(doc, tagset='universal')
            lemmatized_doc = [lemmatizer.lemmatize(token, pos_map.get(tag, 'n')) for token, tag in doc_with_tag]
            result_corpus.append(lemmatized_doc)
        return result_corpus
    def remove_tokens(self, corpus, tokens_to_remove, remove_single_char=False, print_process=False):
        tokens_to_remove = set(tokens_to_remove)
        cleaned_corpus = []
        for doc in corpus:
            cleaned_doc = [w for w in doc if (w not in tokens_to_remove) and ((not remove_single_char) or len(w) >= 2)]
            cleaned_corpus.append(cleaned_doc)
        return cleaned_corpus
    def add_collocations(self, corpus, collocations, print_process=False):
        result_corpus = []
        for doc in corpus:
            doc = ' '.join(doc)
            for collocation in collocations:
                collocation_with_space = collocation.replace('-', ' ')
                doc = doc.replace(collocation_with_space, collocation)
            doc = doc.split(' ')
            result_corpus.append(doc)
        return result_corpus
    def text_preprocessing(self, texts):
        corpus = self.tokenize(texts)
        with open('collocations.txt', 'r') as f:
            collocations = set(w.strip().lower() for w in f if w.strip())
        corpus = self.add_collocations(corpus, collocations)
        with open('typos.txt', 'r') as f:
            typos_dict = {line.split(':')[0]: line.split(':')[1].strip() for line in f}
        corpus = [[typos_dict.get(token, token) for token in doc] for doc in corpus]
        with open('removed_tokens.txt', 'r') as f:
            removed_tokens = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, removed_tokens)
        corpus = self.lemmatize(corpus)
        with open("stopwords_en.txt", "r", encoding="utf-8") as f:
            stop_words = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, stop_words, remove_single_char=True)
        processed_texts = [' '.join(doc) for doc in corpus]
        return processed_texts
    def calc_weighted_vectors(self, df, attribute, vocab_dict, model):
        tfidf_matrix = self.tfidf_vectorizer.transform(df[attribute].fillna(''))
        embedding_matrix = np.zeros((len(vocab_dict), model.wv.vector_size))
        for token, idx in vocab_dict.items():
            if token in model.wv.key_to_index:
                embedding_matrix[idx] = model.wv[token]
        weighted_vectors = []
        for doc_idx in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(doc_idx)
            indices = row.indices
            weights = row.data
            if len(indices) == 0:
                weighted_vectors.append(np.zeros(model.wv.vector_size))
                continue
            word_vecs = embedding_matrix[indices]
            weighted_sum = np.dot(weights, word_vecs)
            weighted_avg = weighted_sum / weights.sum()
            weighted_vectors.append(weighted_avg)
        return np.vstack(weighted_vectors)

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
# DB Models
# -------------------------
class Item(db.Model):
    __tablename__ = "items"
    id = db.Column(db.Integer, primary_key=True)
    source_clothing_id = db.Column(db.Integer, unique=True)
    title = db.Column(db.String, nullable=False)
    description = db.Column(db.Text)
    class_name = db.Column(db.String)
    department_name = db.Column(db.String)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (db.Index("idx_items_title", "title"),)
    reviews = db.relationship("Review", backref="item", lazy=True, cascade="all, delete-orphan")

class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey("items.id", ondelete="CASCADE"), nullable=False)
    reviewer_age = db.Column(db.Integer)
    title = db.Column(db.String)
    body = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    recommend_label = db.Column(db.Integer, nullable=False)
    model_suggested = db.Column(db.Integer, nullable=False, default=0)
    positive_feedback_count = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (
        db.CheckConstraint("rating BETWEEN 1 AND 5", name="chk_rating_range"),
        db.CheckConstraint("recommend_label IN (0,1)", name="chk_rec_label"),
        db.Index("idx_reviews_item_created", "item_id", "created_at"),
    )

# -------------------------
# Model loader (expects a text→label Pipeline)
# -------------------------
pipeline = None
if os.path.exists(MODEL_P):
    try:
        pipeline = joblib.load(MODEL_P)
        print("[Model] Loaded:", MODEL_P)
        try:
            pipeline.predict(["warm up text"])
        except Exception:
            pass
    except Exception as e:
        print("[Model] Failed to load:", e)
else:
    print("[Model] File not found:", MODEL_P)

class ModelUnavailable(Exception): pass

def predict_label_strict(text: str) -> int:
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
# Search indexes (simple + TF-IDF)
# -------------------------
INVERTED = defaultdict(set)
TOKENS_PER_ITEM = {}  # item_id -> Counter(tokens)
WEIGHTS = {"title": 3.0, "class": 2.0, "department": 2.0, "description": 1.0}
STEMMER = PorterStemmer() if PorterStemmer else None

TFIDF_VECT = None
TFIDF_MAT  = None
TFIDF_IDS  = []

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

def build_tfidf_index():
    global TFIDF_VECT, TFIDF_MAT, TFIDF_IDS
    TFIDF_VECT = TFIDF_MAT = None
    TFIDF_IDS = []

    try:
        items = Item.query.all()
    except Exception:
        items = []

    if not items:
        print("[Search][TFIDF] No items to index.")
        return

    docs, ids = [], []
    for it in items:
        title = (it.title or "").strip()
        cls   = (it.class_name or "").strip()
        dept  = (it.department_name or "").strip()
        desc  = (it.description or "").strip()
        doc = " ".join([
            (title + " ") * 2,
            (cls + " ") * 3,
            (dept + " ") * 2,
            desc,
        ])
        docs.append(doc)
        ids.append(it.id)

    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)
        mat  = vect.fit_transform(docs)
        TFIDF_VECT, TFIDF_MAT, TFIDF_IDS = vect, mat, ids
        print(f"[Search][TFIDF] Indexed {len(ids)} items, vocab={len(vect.vocabulary_)}")
    except Exception as e:
        TFIDF_VECT = TFIDF_MAT = None
        TFIDF_IDS = []
        print("[Search][TFIDF] Failed to build:", e)

def build_index():
    INVERTED.clear()
    TOKENS_PER_ITEM.clear()
    for item in Item.query.all():
        index_item(item)
    print(f"[Search] Indexed {len(TOKENS_PER_ITEM)} items, {len(INVERTED)} tokens")
    build_tfidf_index()

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

def rank_items(query: str, mode: str = "simple"):
    q = (query or "").strip()
    if not q:
        return [], "simple"

    if mode == "tfidf":
        if TFIDF_VECT is None or TFIDF_MAT is None:
            build_tfidf_index()
        if TFIDF_VECT is not None and TFIDF_MAT is not None:
            try:
                qvec = TFIDF_VECT.transform([q])
                sims = (qvec @ TFIDF_MAT.T).toarray().ravel()
                order = sims.argsort()[::-1]
                ids = [TFIDF_IDS[i] for i in order if sims[i] > 0]
                return ids, "tfidf"
            except Exception as e:
                print("[Search][TFIDF] query failed:", e)
        print("[Search][TFIDF] unavailable; fallback to simple")
        return score_items(q), "simple"

    return score_items(q), "simple"

# -------------------------
# CSV import / bootstrap
# -------------------------
def load_items_from_csv(path: str = DATA_CSV) -> int:
    if not os.path.exists(path):
        print(f"[Import] CSV not found at {path}")
        return 0

    df = pd.read_csv(path)

    # normalize headers
    def norm(s): return str(s).strip().lower().replace("_", " ")
    cols = {norm(c): c for c in df.columns}
    def col(*names):
        for n in names:
            k = norm(n)
            if k in cols: return cols[k]
        return None

    c_id    = col("clothing id", "clothing_id", "id")
    c_title = col("clothes title", "item title", "product title", "title")
    c_desc  = col("clothes description", "description", "product description")
    c_class = col("class name", "class")
    c_dept  = col("department name", "department")

    if not c_id:
        print("[Import] ERROR: 'Clothing ID' column not found — cannot create real items.")
        return 0

    # one row per unique Clothing ID
    df = df[df[c_id].notna()].copy()
    df[c_id] = df[c_id].astype(int)
    df = df.sort_values(by=[c_id]).drop_duplicates(subset=[c_id], keep="first")

    added = 0
    for _, r in df.iterrows():
        srcid = int(r[c_id])
        title = str(r.get(c_title, "") or "").strip()
        desc  = str(r.get(c_desc, "") or "").strip()
        cls   = str(r.get(c_class, "") or "").strip()
        dept  = str(r.get(c_dept, "") or "").strip()

        item = Item.query.filter_by(source_clothing_id=srcid).first()
        if not item:
            item = Item(
                source_clothing_id=srcid,
                title=title or f"Item {srcid}",
                description=desc,
                class_name=cls,
                department_name=dept,
            )
            db.session.add(item)
            added += 1
        else:
            # fill missing fields if CSV has them
            if not item.title and title: item.title = title
            if not item.description and desc: item.description = desc
            if not item.class_name and cls: item.class_name = cls
            if not item.department_name and dept: item.department_name = dept

    db.session.commit()
    print(f"[Import] Items in DB: {Item.query.count()} (+{added} added)")
    return added

def load_reviews_from_csv(path: str = DATA_CSV, limit: int | None = None) -> int:
    """
    Import reviews from the CSV and attach them to items.
    Expected columns:
      'Clothing ID', 'Title', 'Review Text', 'Rating', 'Recommended IND',
      'Positive Feedback Count', 'Age', 'Clothes Title'
    """
    if not os.path.exists(path):
        print(f"[Import][Reviews] CSV not found at {path}")
        return 0

    df = pd.read_csv(path)

    # Build lookups
    id_map = {it.source_clothing_id: it.id for it in Item.query.all() if it.source_clothing_id is not None}
    title_map = {it.title.strip().lower(): it.id for it in Item.query.all() if it.title}

    # Dedupe key: (item_id, body[:120], title[:80])
    existing_keys = set(
        (rv.item_id, (rv.body or "")[:120], (rv.title or "")[:80])
        for rv in Review.query.all()
    )

    added = 0
    for _, row in df.iterrows():
        if limit and added >= limit:
            break

        # Resolve item_id
        item_id = None
        srcid = row.get("Clothing ID", None)
        if pd.notna(srcid):
            try:
                srcid = int(srcid)
                item_id = id_map.get(srcid)
            except Exception:
                item_id = None
        if not item_id:
            ctitle = str(row.get("Clothes Title", "") or "").strip().lower()
            if ctitle:
                item_id = title_map.get(ctitle)
        if not item_id:
            continue

        rev_title = str(row.get("Title", "") or "").strip()
        body = str(row.get("Review Text", "") or "").strip()
        if not body:
            continue

        try:
            rating = int(row.get("Rating", 5))
        except Exception:
            rating = 5
        rating = max(1, min(5, rating))

        try:
            rec = int(row.get("Recommended IND", 0))
            rec = 1 if rec == 1 else 0
        except Exception:
            rec = 0

        try:
            age = int(row.get("Age"))
        except Exception:
            age = None

        try:
            pfc = int(row.get("Positive Feedback Count", 0))
        except Exception:
            pfc = 0

        key = (item_id, body[:120], rev_title[:80])
        if key in existing_keys:
            continue

        rv = Review(
            item_id=item_id,
            title=rev_title if rev_title else None,
            body=body,
            rating=rating,
            recommend_label=rec,
            model_suggested=rec,  # historical data -> mirror label
            positive_feedback_count=pfc,
            reviewer_age=age,
        )
        db.session.add(rv)
        existing_keys.add(key)
        added += 1

        if added % 1000 == 0:
            db.session.flush()

    db.session.commit()
    print(f"[Import][Reviews] Added {added} reviews.")
    return added

def bootstrap_if_needed():
    db.create_all()
    if Item.query.count() == 0:
        print("[Bootstrap] Items empty; importing CSV…")
        load_items_from_csv()
    if Review.query.count() == 0:
        print("[Bootstrap] Reviews empty; importing CSV reviews…")
        load_reviews_from_csv()
    build_index()

with app.app_context():
    bootstrap_if_needed()


def items_with_rec_order(ids: list[int] | None = None, limit: int | None = None):
    """
    Return items ordered by SUM(Review.recommend_label) DESC,
    then by total review count (DESC), then by item id.
    Also returns a small stats map so templates can show badges.
    """
    q = db.session.query(
        Item,
        db.func.coalesce(db.func.sum(Review.recommend_label), 0).label("rec_sum"),
        db.func.count(Review.id).label("review_count"),
    ).outerjoin(Review, Review.item_id == Item.id)

    if ids:
        q = q.filter(Item.id.in_(ids))

    q = (
        q.group_by(Item.id)
         .order_by(db.desc("rec_sum"), db.desc("review_count"), Item.id)
    )

    if limit:
        q = q.limit(limit)

    rows = q.all()
    items = [row[0] for row in rows]
    rec_map = {row[0].id: {"rec_sum": int(row[1] or 0), "review_count": int(row[2] or 0)} for row in rows}
    return items, rec_map


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    mode = request.args.get("mode", "simple")  # 'simple' or 'tfidf'
    used_mode = mode
    count = None

    if q:
        ids, used_mode = rank_items(q, mode)  # filter by query
        count = len(ids)
        items, rec_map = items_with_rec_order(ids=ids)  # sort by recommended count
    else:
        items, rec_map = items_with_rec_order(limit=24)  # top recommended on homepage

    return render_template("index.html", items=items, q=q, count=count, mode=used_mode, rec_map=rec_map)

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

        try:
            suggested = predict_label_strict((title + " " + body).strip())
            print(f"[Model] Final submit predicted {suggested} for: {title} {body}")
        except Exception:
            flash("Prediction unavailable right now. Please try again.", "danger")
            return redirect(request.url)

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
    title = request.form.get("title", "").strip()
    body  = request.form.get("body", "").strip()
    text  = (title + " " + body).strip()
    try:
        lbl = predict_label_strict(text)
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
    """
    Import/merge items and reviews from the CSV.
    Params:
      - wipe=1  : delete all items+reviews first
      - limit=N : only import first N reviews (helpful for testing)
    """
    if request.args.get("wipe") == "1":
        Review.query.delete()
        Item.query.delete()
        db.session.commit()
        print("[Import] Wiped items + reviews")

    added_items = load_items_from_csv()
    limit = request.args.get("limit", type=int)
    added_reviews = load_reviews_from_csv(limit=limit)

    build_index()
    flash(f"Imported CSV · items +{added_items}, reviews +{added_reviews}. Index rebuilt.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
