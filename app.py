import os
import re
import math
from collections import defaultdict, Counter
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np

# Optional NLP deps (safely imported)
try:
    from nltk.stem import PorterStemmer
except Exception:  # nltk may be missing in minimal environments
    PorterStemmer = None

# Heavier optional deps used only in advanced preprocessing (guarded)
try:
    import html
    import nltk
    from nltk.tokenize import RegexpTokenizer, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from bs4 import BeautifulSoup
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_extraction.text import TfidfVectorizer
    from gensim.models import FastText
except Exception:  # keep app running even if these aren't installed
    html = None
    nltk = None
    RegexpTokenizer = None
    sent_tokenize = None
    WordNetLemmatizer = None
    BeautifulSoup = None
    BaseEstimator = None
    TransformerMixin = None
    TfidfVectorizer = None
    FastText = None


class TextToVectorTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that handles text preprocessing and
    converts it to weighted FastText vectors.
    """
    def __init__(self):
        self.vocab = None
        self.fasttext_model = None
        self.tfidf_vectorizer = None
        self.data_is_preprocessed = False

    def fit(self, X, y = None):
        """
        Fits the transformer by preprocessing the data and training the FastText model.
        
        Args:
            X (List[str]): The input text data.
            y (List[int], optional): The target variable. Not used in this transformer.
        """
        # Retrieve cleaned text and vocab (already processed in task 1)
        self.vocab = read_vocab('vocab_both.txt')
        
        # Train the FastText model on the preprocessed data
        self.fasttext_model = FastText.load('fasttext_model.model')
        
        # Fit the weighted vectorizer (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', vocabulary=self.vocab, lowercase=True)
        self.tfidf_vectorizer.fit(X)
        
        self.data_is_preprocessed = True
        
        return self

    def transform(self, X):
        """
        Transforms new data using the fitted preprocessing steps and model.
        
        Args:
            X (List[str]): The new input text data.

        Returns:
            np.ndarray: The weighted vectors for the input data.
        """
        if self.vocab is None or self.fasttext_model is None or self.tfidf_vectorizer is None:
            print(self.vocab, self.fasttext_model, self.tfidf_vectorizer)
            raise RuntimeError("This transformer has not been fitted yet. Call .fit() before .transform().")
        
        # Generate weighted vectors using the stored vocab and model
        data = pd.DataFrame({
            'New Review': X if self.data_is_preprocessed else self.text_preprocessing(X)
        })
        weighted_vectors = self.calc_weighted_vectors(data, 'New Review', self.vocab, self.fasttext_model)
        self.data_is_preprocessed = False

        return weighted_vectors
    
    # Function to tokenize pd.DataFrame to corpus (a 2D list, each row stores tokens of a review)
    def tokenize(self, texts, get_vocab = False, print_process = False):
        '''
        Perform sentence segmentation and word tokenization.

        Args:
            df (pd.DataFrame): DataFrame containing text column
            attribute (str): Name of text column
            print_process (bool): Print the result of the tokenization process or not

        Returns:
            corpus (list of list): 2D list of tokens per row
        '''
        regex_tokenizer = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        ENTITY_RE = re.compile(r"&(?:[A-Za-z]+|#[0-9]+|#x[0-9A-Fa-f]+);")
        corpus = []

        for text in texts:
            tokens = []

            # drop HTML encode
            unescaped = html.unescape(text)
            soup = BeautifulSoup(unescaped, 'html.parser')
            text = soup.get_text()
            text = ENTITY_RE.sub(' ', text)

            # tokenization
            for sent in sent_tokenize(text): # sentence segmentation
                words = regex_tokenizer.tokenize(sent) # word tokenization
                words = [w.lower() for w in words] # lowercase transform
                tokens.extend(words)
            corpus.append(tokens)

        # Build vocab dict (alphabetical order)
        if get_vocab:
            unique_tokens = sorted({t for doc in corpus for t in doc})
            vocab = {token: idx for idx, token in enumerate(unique_tokens)}
            return corpus, vocab

        # Print process
        if print_process:
            print(f"Finish tokenize: {sum([len(tokens) for tokens in corpus])} token extracted")

        return corpus

    # Lemmatization
    def lemmatize(self, corpus, print_process = False):
        '''
        Apply lemmatization to 2D token list.

        Args:
            corpus (list of list)

        Returns:
            corpus (list of list): lemmatized tokens
        '''
        result_corpus = []
        pos_map = {
            'ADJ': 'a',
            'ADP': 's',
            'ADV': 'r',
            'NOUN': 'n', # assume any undefined tags (like DET, PRON, ...) is n (NOUN)
            'VERB': 'v',
        }

        lemmatizer = WordNetLemmatizer()
        for doc in corpus:
            doc_with_tag = nltk.pos_tag(doc, tagset = 'universal') # set POS tag for all tokens in doc (tag is the type of word: NOUN, ADJ, ...)
            lemmatized_doc = [lemmatizer.lemmatize(token, pos_map.get(tag, 'n')) for token, tag in doc_with_tag] # assume any undefined tags (like DET, PRON, ...) is n (NOUN)
            result_corpus.append(lemmatized_doc)

        # Print process
        if print_process:
            print("Finish lemmatize:")
            print(f"+ Before: {sum([len(review_tokens) for review_tokens in corpus])} tokens: {corpus[:5]} ...")
            print(f"+ Now:    {sum([len(review_tokens) for review_tokens in result_corpus])} tokens: {result_corpus[:5]} ...")
        return result_corpus

    # Function to remove invalid tokens
    def remove_tokens(self, corpus, tokens_to_remove, remove_single_char = False, print_process = False):
        '''
        Remove the tokens of corpus that are in tokens_to_remove

        Args:
            corpus (list of list): tokenized text
            tokens_to_remove (list): list of tokens (str)
            remove_single_char (bool): whether removing tokens with length = 1 or not

        Returns:
            corpus (list of list): cleaned tokenized text
        '''

        tokens_to_remove = set(tokens_to_remove)
        cleaned_corpus = []
        for doc in corpus:
            cleaned_doc = [w for w in doc if (w not in tokens_to_remove) and ((not remove_single_char) or len(w) >= 2)]
            cleaned_corpus.append(cleaned_doc)

        # Print process
        if print_process:
            print("Finish removal:")
            print(f"+ Before: {sum([len(review_tokens) for review_tokens in corpus])} tokens: {corpus[:5]} ...")
            print(f"+ Now:    {sum([len(review_tokens) for review_tokens in cleaned_corpus])} tokens: {cleaned_corpus[:5]} ...")

        return cleaned_corpus
    
    # Function to add collocations to corpus
    def add_collocations(self, corpus, collocations, print_process = False):
        '''
        Add collocations to the corpus

        Args:
            corpus (list of list): 2D token list
            collocations_dict (dict): Dictionary of top collocations by n-gram type

        Returns:
            corpus (list of list): Corpus with detected collocations treated as single tokens (e.g., 'new-york')
            replaced_tokens (dict): All tokens that have been replaced by collocations in a form {token_be_replaced: collocation}
        '''
        result_corpus = []

        for doc in corpus:
            doc = ' '.join(doc) # doc is transfromed from ['he', 'work', 'out', ...] to 'he work out ...' so all collocations will be separated with space
            for collocation in collocations:
                collocation_with_space = collocation.replace('-', ' ')
                doc = doc.replace(collocation_with_space, collocation) # replace all collocation with space to collocation with "-", like 'work out' to 'work-out'
            doc = doc.split(' ') # doc is transformed to ['he', 'work-out', ...]
            result_corpus.append(doc)

        if print_process:
            print(f"Finish add collocations:")
            print(f"+ Before: {sum([len(review_tokens) for review_tokens in corpus])} tokens: {corpus[:5]} ...")
            print(f"+ Now:    {sum([len(review_tokens) for review_tokens in result_corpus])} tokens: {result_corpus[:5]} ...")

        return result_corpus

    # Function to implement full pipeline
    def text_preprocessing(self, texts):
        '''
        Idea:
            1. Remove all tokens that were removed in training dataset (task 1)
            2. Replace all tokens that were replaced in training dataset. E.g: if in training dataset, "rcm" is replaced by "recommend", so in new text, "rcm" is also be replaced similar. 
        Args:
            texts (List[str]): List of all texts that need to be preprocessing.
        Returns:
            processed_texts (List[str]): List of all texts after preprocessing.
        '''
        # ---- Sentence segmentation -> word tokenization ----
        corpus = self.tokenize(texts)

        # ---- Handle collocations and typos ----
        with open('collocations.txt', 'r') as f:
            collocations = set(w.strip().lower() for w in f if w.strip())
        corpus = self.add_collocations(corpus, collocations)
        with open('typos.txt', 'r') as f:
            typos_dict = {line.split(':')[0]: line.split(':')[1].strip() for line in f}
        corpus = [[typos_dict.get(token, token) for token in doc] for doc in corpus]

        # ---- Text removal ----
        with open('removed_tokens.txt', 'r') as f:
            removed_tokens = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, removed_tokens)

        # ---- Lemmatization ----
        corpus = self.lemmatize(corpus)
        with open("stopwords_en.txt", "r", encoding="utf-8") as f: # Download stopwords_en.txt
            stop_words = set(w.strip().lower() for w in f if w.strip())
        corpus = self.remove_tokens(corpus, stop_words, remove_single_char = True) # Text removal after lemmatization: stopwords + tokens with length = 1

        # ---- Output ----
        processed_texts = [' '.join(doc) for doc in corpus]
        return processed_texts
    # Function to calculate weighted vectors (document representation) based on an embedding model loaded in advance
    def calc_weighted_vectors(self, df, attribute, vocab_dict, model):
        '''
        Calculates TF-IDF weighted document vectors.

        Args:
            df: The DataFrame containing the text data.
            attribute: The column name in the DataFrame with the text.
            vocab_dict: A dictionary mapping vocabulary tokens to their unique IDs.
            model: The pre-trained word embedding model (e.g., Word2Vec, FastText).

        Returns:
            numpy.ndarray: A 2D array where each row is the weighted vector for a document.
        '''
        # Use TfidfVectorizer with the predefined vocabulary to get TF-IDF scores
        tfidf_matrix = self.tfidf_vectorizer.transform(df[attribute].fillna(''))

        # Precompute embedding matrix aligned with vocab_dict
        embedding_matrix = np.zeros((len(vocab_dict), model.wv.vector_size))
        for token, idx in vocab_dict.items():
            if token in model.wv.key_to_index:  # Check if token exists in pretrained model
                embedding_matrix[idx] = model.wv[token]
            # else remains zero vector

        # Compute Weighted Review Vectors (TF-IDF weighted mean)
        weighted_vectors = []
        for doc_idx in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(doc_idx)
            indices = row.indices # only get element that is not = 0
            weights = row.data

            if len(indices) == 0:
                weighted_vectors.append(np.zeros(model.wv.vector_size))
                continue

            # Get the corresponding word vectors from the precomputed embedding matrix
            word_vecs = embedding_matrix[indices]

            # Perform a dot product to get the weighted sum
            weighted_sum = np.dot(weights, word_vecs)
            weighted_avg = weighted_sum / weights.sum()
            weighted_vectors.append(weighted_avg)

        return np.vstack(weighted_vectors)  # shape: (n_docs, vector_size)
# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "shop.db")
DATA_CSV = os.path.join(BASE_DIR, "data", "clothing_reviews_m2.csv")
MODEL_P = os.path.join(BASE_DIR, "models", "review_recommender.joblib")

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
    recommend_label = db.Column(db.Integer, nullable=False)  # 0/1
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
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


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
        # If the pipeline supports probabilities directly
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([text])[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Or supports decision_function → map with sigmoid
        if hasattr(pipeline, "decision_function"):
            score = float(pipeline.decision_function([text])[0])
            return float(_sigmoid(score))

        # Fallback to plain predict then map to 0/1
        pred = int(pipeline.predict([text])[0])
        return 1.0 if pred == 1 else 0.0

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
        try:
            return STEMMER.stem(w)
        except Exception:
            pass
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


def index_item(item):
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
        c = TOKENS_PER_ITEM.get(item_id, Counter())
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
        else:
            df = pd.read_csv(DATA_CSV)
            # Expected columns:
            #  'Clothes Title', 'Clothes Description', 'Class Name', 'Department Name'
            # If names differ, adjust here.
            rows = []
            for _, r in df.iterrows():
                title = str(r.get("Clothes Title", "")).strip()
                desc = str(r.get("Clothes Description", "")).strip()
                cls = str(r.get("Class Name", "")).strip()
                dept = str(r.get("Department Name", "")).strip()
                if not title:
                    continue
                rows.append(
                    Item(
                        title=title,
                        description=desc,
                        class_name=cls,
                        department_name=dept,
                    )
                )
            if rows:
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
        if ids:
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
    reviews = (
        Review.query.filter_by(item_id=item.id)
        .order_by(Review.created_at.desc())
        .all()
    )
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

        # ⚡ Compute model suggestion now
        text = (title + " " + body).strip()
        prob = predict_recommend_proba(text)  # returns 0..1
        suggested = 1 if prob >= 0.5 else 0

        # User can override via dropdown
        final_lbl = int(request.form.get("recommend_label", suggested))

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

    # GET → render the form (JS will call /suggest as the user types)
    return render_template("review_form.html", item=item, suggested=None)
    item = Item.query.get_or_404(item_id)
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        body = request.form.get("body", "").strip()
        rating = int(request.form.get("rating", "5"))
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
    title = request.form.get("title","").strip()
    body  = request.form.get("body","").strip()
    text  = (title + " " + body).strip()
    prob = predict_recommend_proba(text) if text else 0.5
    lbl = 1 if prob >= 0.5 else 0
    return {"label": lbl, "prob": round(prob, 3)}


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
