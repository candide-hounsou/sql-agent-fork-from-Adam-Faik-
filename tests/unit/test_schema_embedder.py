"""Unit tests for the TF-IDF schema embedder (Phase 3 — RAG schema linking)."""

import os
import pickle

SAMPLE_SCHEMA = """\
CREATE TABLE customers (
    customer_id TEXT,
    customer_city TEXT,
    customer_state TEXT
);

CREATE TABLE orders (
    order_id TEXT,
    customer_id TEXT,
    order_status TEXT
);

CREATE TABLE products (
    product_id TEXT,
    product_category_name TEXT,
    product_weight_g REAL
);

CREATE TABLE order_items (
    order_id TEXT,
    product_id TEXT,
    price REAL
);
"""


class TestBuildSchemaIndex:
    def test_creates_pickle_file(self, tmp_path):
        from src.schema.embedder import build_schema_index

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)
        assert os.path.exists(index_path + ".pkl")

    def test_pickle_contains_expected_keys(self, tmp_path):
        from src.schema.embedder import build_schema_index

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)

        with open(index_path + ".pkl", "rb") as f:
            data = pickle.load(f)

        assert "blocks" in data
        assert "vectorizer" in data
        assert "matrix" in data

    def test_blocks_count_matches_tables(self, tmp_path):
        from src.schema.embedder import build_schema_index

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)

        with open(index_path + ".pkl", "rb") as f:
            data = pickle.load(f)

        # Four CREATE TABLE blocks in SAMPLE_SCHEMA
        assert len(data["blocks"]) == 4

    def test_empty_schema_does_not_crash(self, tmp_path):
        from src.schema.embedder import build_schema_index

        index_path = str(tmp_path / "schema_index")
        build_schema_index("", index_path=index_path)
        # No file should be written for empty schema
        assert not os.path.exists(index_path + ".pkl")


class TestRetrieveRelevantSchema:
    def test_returns_relevant_table(self, tmp_path):
        from src.schema.embedder import build_schema_index, retrieve_relevant_schema

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)

        result = retrieve_relevant_schema("customer city state", index_path=index_path, top_k=1)
        assert "customers" in result.lower()

    def test_returns_multiple_tables(self, tmp_path):
        from src.schema.embedder import build_schema_index, retrieve_relevant_schema

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)

        result = retrieve_relevant_schema("order product price", index_path=index_path, top_k=3)
        # Should include at least 2 of the 3 relevant tables
        assert result.count("CREATE TABLE") >= 2

    def test_fallback_to_full_schema_when_index_missing(self, tmp_path, monkeypatch):
        from src.schema.embedder import retrieve_relevant_schema

        # Point to a non-existent index
        missing_path = str(tmp_path / "missing_index")

        # Monkeypatch open to simulate data/schema.txt content
        schema_content = "CREATE TABLE fallback_table (id INTEGER);"
        schema_file = tmp_path / "schema.txt"
        schema_file.write_text(schema_content, encoding="utf-8")

        original_open = open

        def mock_open(path, *args, **kwargs):
            if path == "data/schema.txt":
                return original_open(str(schema_file), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)
        result = retrieve_relevant_schema("anything", index_path=missing_path, top_k=3)
        assert "fallback_table" in result

    def test_top_k_limits_results(self, tmp_path):
        from src.schema.embedder import build_schema_index, retrieve_relevant_schema

        index_path = str(tmp_path / "schema_index")
        build_schema_index(SAMPLE_SCHEMA, index_path=index_path)

        result = retrieve_relevant_schema("data", index_path=index_path, top_k=2)
        assert result.count("CREATE TABLE") <= 2
