"""Unit tests for src/storage/history.py (Phase 8 — persistent sessions)."""

import os

import pytest


class TestSessionHistory:
    def test_new_creates_unique_ids(self):
        from src.storage.history import SessionHistory

        s1 = SessionHistory.new()
        s2 = SessionHistory.new()
        assert s1.session_id != s2.session_id

    def test_new_default_name(self):
        from src.storage.history import SessionHistory

        s = SessionHistory.new()
        assert s.name == "New Session"

    def test_new_custom_name(self):
        from src.storage.history import SessionHistory

        s = SessionHistory.new(name="My Test Session")
        assert s.name == "My Test Session"

    def test_new_has_iso_timestamp(self):
        from datetime import datetime

        from src.storage.history import SessionHistory

        s = SessionHistory.new()
        # Should parse without raising
        dt = datetime.fromisoformat(s.created_at)
        assert dt is not None

    def test_messages_default_empty(self):
        from src.storage.history import SessionHistory

        s = SessionHistory.new()
        assert s.messages == []


class TestHistoryStore:
    @pytest.fixture
    def store(self, tmp_path):
        from src.storage.history import HistoryStore

        return HistoryStore(store_path=str(tmp_path / "history.json"))

    def test_save_and_load_roundtrip(self, store):
        from src.storage.history import SessionHistory

        session = SessionHistory.new(name="Test Session")
        session.messages = [{"role": "user", "content": "hello"}]
        store.save_session(session)

        loaded = store.load_session(session.session_id)
        assert loaded is not None
        assert loaded.name == "Test Session"
        assert loaded.messages == [{"role": "user", "content": "hello"}]

    def test_load_nonexistent_returns_none(self, store):
        assert store.load_session("does-not-exist") is None

    def test_list_sessions_empty_initially(self, store):
        assert store.list_sessions() == []

    def test_list_sessions_returns_all(self, store):
        from src.storage.history import SessionHistory

        s1 = SessionHistory.new(name="A")
        s2 = SessionHistory.new(name="B")
        store.save_session(s1)
        store.save_session(s2)

        sessions = store.list_sessions()
        names = {s.name for s in sessions}
        assert names == {"A", "B"}

    def test_list_sessions_sorted_newest_first(self, store):
        import time

        from src.storage.history import SessionHistory

        s1 = SessionHistory.new(name="First")
        time.sleep(0.01)
        s2 = SessionHistory.new(name="Second")
        store.save_session(s1)
        store.save_session(s2)

        sessions = store.list_sessions()
        assert sessions[0].name == "Second"

    def test_delete_existing_session(self, store):
        from src.storage.history import SessionHistory

        session = SessionHistory.new()
        store.save_session(session)

        result = store.delete_session(session.session_id)
        assert result is True
        assert store.load_session(session.session_id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_session("ghost-id") is False

    def test_rename_session(self, store):
        from src.storage.history import SessionHistory

        session = SessionHistory.new(name="Old Name")
        store.save_session(session)

        result = store.rename_session(session.session_id, "New Name")
        assert result is True

        loaded = store.load_session(session.session_id)
        assert loaded.name == "New Name"

    def test_concurrent_writes_do_not_corrupt(self, store, tmp_path):
        """Write multiple sessions from different HistoryStore instances."""
        from src.storage.history import HistoryStore, SessionHistory

        path = str(tmp_path / "history.json")
        stores = [HistoryStore(store_path=path) for _ in range(5)]
        sessions = [SessionHistory.new(name=f"S{i}") for i in range(5)]

        for s, st_ in zip(sessions, stores):
            st_.save_session(s)

        # All sessions should be present
        final_store = HistoryStore(store_path=path)
        all_sessions = final_store.list_sessions()
        assert len(all_sessions) == 5

    def test_store_creates_parent_directory(self, tmp_path):
        from src.storage.history import HistoryStore, SessionHistory

        deep_path = str(tmp_path / "a" / "b" / "c" / "history.json")
        store = HistoryStore(store_path=deep_path)
        session = SessionHistory.new()
        store.save_session(session)
        assert os.path.exists(deep_path)
