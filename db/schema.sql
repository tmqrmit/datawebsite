PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS items (
  id                 INTEGER PRIMARY KEY,
  source_clothing_id INTEGER UNIQUE,               -- optional extra
  title              TEXT NOT NULL,
  description        TEXT,
  class_name         TEXT,
  department_name    TEXT,
  created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_items_source ON items(source_clothing_id);

CREATE TABLE IF NOT EXISTS reviews (
  id                      INTEGER PRIMARY KEY,
  item_id                 INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  title                   TEXT,
  body                    TEXT NOT NULL,
  rating                  INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  recommend_label         INTEGER NOT NULL CHECK (recommend_label IN (0,1)),
  model_suggested         INTEGER NOT NULL DEFAULT 0 CHECK (model_suggested IN (0,1)),
  positive_feedback_count INTEGER NOT NULL DEFAULT 0 CHECK (positive_feedback_count >= 0),
  created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reviews_item_created ON reviews(item_id, created_at DESC);
