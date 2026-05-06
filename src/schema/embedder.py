"""Lightweight TF-IDF schema index for RAG-augmented schema linking."""

import os
import pickle
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _parse_table_blocks(schema_text: str) -> List[str]:
    """Split a schema string into individual CREATE TABLE blocks."""
    blocks = re.split(r"(?=CREATE TABLE\s)", schema_text, flags=re.IGNORECASE)
    return [b.strip() for b in blocks if b.strip()]


def _normalize(text: str) -> str:
    """Replace underscores with spaces and lower-case for TF-IDF matching.

    Schema identifiers such as ``customer_city`` become ``customer city``
    so that natural-language queries like "customers by city" match correctly.
    """
    return re.sub(r"_", " ", text).lower()


def build_schema_index(schema_text: str, index_path: str = "data/schema_index") -> None:
    """Build a TF-IDF index from schema table blocks and persist it to disk.

    Parameters
    ----------
    schema_text:
        Full schema string (e.g. from data/schema.txt).
    index_path:
        Base path for the pickle file (``<index_path>.pkl`` is written).
    """
    blocks = _parse_table_blocks(schema_text)
    if not blocks:
        print("⚠️  No CREATE TABLE blocks found in schema — index not built.")
        return

    normalized_blocks = [_normalize(b) for b in blocks]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(normalized_blocks)

    index_data = {
        "blocks": blocks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }

    dir_name = os.path.dirname(index_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    pkl_path = index_path + ".pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"✅ Schema index built: {len(blocks)} table block(s) → {pkl_path}")


def retrieve_relevant_schema(
    query: str, index_path: str = "data/schema_index", top_k: int = 5
) -> str:
    """Return the top-k most relevant table schema blocks for *query*.

    Falls back to the full ``data/schema.txt`` if the index does not exist.
    """
    pkl_path = index_path + ".pkl"
    if not os.path.exists(pkl_path):
        try:
            with open("data/schema.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Schema file not found."

    with open(pkl_path, "rb") as f:
        index_data = pickle.load(f)

    blocks: List[str] = index_data["blocks"]
    vectorizer: TfidfVectorizer = index_data["vectorizer"]
    matrix = index_data["matrix"]

    query_vec = vectorizer.transform([_normalize(query)])
    scores = cosine_similarity(query_vec, matrix).flatten()

    k = min(top_k, len(blocks))
    top_indices = np.argsort(scores)[::-1][:k]

    return "\n\n".join(blocks[i] for i in top_indices)
