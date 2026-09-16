"""recall() must degrade to FTS when the vec index fails, not crash.

The incident: a torn vec0 write left ``vec_episodes_vector_chunks00`` a
0-byte blob while its chunk metadata claimed 1024 valid rows. Every KNN on
``vec_episodes`` raised ``sqlite3.OperationalError: vectors blob size doesn't
match - expected 1048576, found 0``, which propagated out of
``BeamMemory.recall()`` and made the FTS/keyword branch unreachable. Dense
recall died 100%, and the failure surfaced only as generic prefetch timeouts
in the host agent.

Covered here:

1. A failing KNN still returns FTS results, and logs the real sqlite-vec
   error text plus the fallback consequence (``test_recall_falls_back_to_
   fts_when_vec_search_raises``).
2. The per-connection failure latch skips the known-bad KNN on later calls
   instead of re-running (and re-logging) it every time
   (``test_recall_vec_latch_skips_repeated_knn``).
3. The latch clears when another connection commits -- the out-of-process
   repair shape (``test_recall_vec_latch_cleared_by_data_version``).
4. A healthy index is unchanged: dense results still served.
5. Episodic vec-insert failures are logged at ERROR with table + rowid and a
   one-time reindex hint, while the memory row survives
   (``test_consolidate_to_episodic_persists_row_and_logs_error_on_vec_failure``).
6. ``_refresh_episodic_embedding`` degrades on a sqlite3.Error instead of
   aborting the content refresh
   (``test_refresh_episodic_embedding_degrades_on_vec_failure``).

The ``_vec_search`` degrade tests live in
``tests/test_vec_search_mismatch_degrade.py``; that helper's strict
classify-then-propagate contract is unchanged by this card.
"""
from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as beam

BLOB_MISMATCH = (
    "vectors blob size doesn't match - expected 1048576, found 0"
)
TOKEN = "zanzibarian"
CONTENT = f"the {TOKEN} quokka migration notes"


@pytest.fixture(autouse=True)
def _clear_latch_state():
    """Keep the module-level latch/hint globals test-local.

    Both are process-global by design (mirroring the pre-existing
    ``_vec_working_count_cache`` pattern), so a test that latches must not
    leak the entry into the next one.
    """
    beam._RECALL_VEC_LATCH.clear()
    beam._VEC_WRITE_HINTED.clear()
    yield
    beam._RECALL_VEC_LATCH.clear()
    beam._VEC_WRITE_HINTED.clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real BeamMemory store with one FTS-reachable episodic row.

    Embeddings are stubbed content-free (the vec voice must be reached, not
    merely absent) and the dimension is pinned to the test store's own so the
    fixture is independent of the runner's environment.
    """
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    db = tmp_path / "recall_degrade.db"
    beam.init_beam(db)

    mem = beam.BeamMemory(session_id="s", db_path=db)
    monkeypatch.setattr(beam._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam._embeddings,
        "embed",
        lambda texts: np.stack([np.full(384, 0.01, dtype=np.float32) for _ in texts]),
    )
    monkeypatch.setattr(
        beam._embeddings,
        "embed_query",
        lambda q: np.full(384, 0.01, dtype=np.float32),
    )

    rowid = mem.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES ('em-1', ?, 'test', datetime('now'), 's', 0.9)",
        (CONTENT,),
    ).lastrowid
    beam._vec_insert(mem.conn, rowid, [0.01] * 384)
    mem.conn.commit()
    return mem


def _error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_recall_falls_back_to_fts_when_vec_search_raises(store, monkeypatch, caplog):
    """A mocked vec KNN failure must not escape recall(): FTS still serves
    the query, and the log carries both the real sqlite-vec error text and
    the fallback consequence in one place."""
    calls = []

    def boom(conn, embedding, k=20):
        calls.append(k)
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_search", boom)

    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        results = store.recall(TOKEN, top_k=5)

    assert calls, "the vec branch must actually have been attempted"
    assert results, "recall must still return the FTS/keyword results"
    assert any(TOKEN in (r.get("content") or "") for r in results), results
    logged = " ".join(r.message for r in _error_records(caplog))
    # The original diagnostic (blob sizes) AND the consequence, so an
    # operator grepping for either finds the same line.
    assert BLOB_MISMATCH in logged, caplog.text
    assert "falling back to FTS" in logged, caplog.text
    assert "mnemosyne reindex" in logged, caplog.text


def test_recall_healthy_index_still_serves_dense_results(store, caplog):
    """The wrap must not weaken a healthy index: the vec voice still runs
    and the row comes back with a dense score, error-free."""
    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        results = store.recall(TOKEN, top_k=5)

    assert results, results
    assert not _error_records(caplog), [r.message for r in caplog.records]
    assert beam._RECALL_VEC_LATCH == {}, "a healthy call must not latch"


def test_recall_vec_latch_skips_repeated_knn(store, monkeypatch, caplog):
    """Second call in the same process must skip the known-bad KNN: one
    failing query and one ERROR per process, not per call."""
    calls = []

    def boom(conn, embedding, k=20):
        calls.append(k)
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_search", boom)

    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        first = store.recall(TOKEN, top_k=5)
    first_errors = len(_error_records(caplog))

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="mnemosyne.core.beam"):
        second = store.recall(TOKEN, top_k=5)

    assert first and second, "both calls must still return FTS results"
    assert len(calls) == 1, f"KNN re-attempted after latching: {calls}"
    assert first_errors >= 1
    assert not _error_records(caplog), [r.message for r in caplog.records]
    # The skip is reported, just not as an error.
    assert any(
        "latched off" in r.message and r.levelno == logging.DEBUG
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_recall_vec_latch_cleared_by_data_version(store, monkeypatch, tmp_path):
    """An out-of-process repair commits through a SECOND connection; the
    next recall() must re-attempt the KNN rather than stay latched off."""
    calls = []

    def boom(conn, embedding, k=20):
        calls.append(k)
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_search", boom)

    store.recall(TOKEN, top_k=5)
    assert len(calls) == 1

    # A separate connection committing is what bumps this connection's
    # PRAGMA data_version (its own commits do not).
    other = sqlite3.connect(str(tmp_path / "recall_degrade.db"))
    try:
        other.execute("CREATE TABLE latch_probe (x INTEGER)")
        other.commit()
    finally:
        other.close()

    store.recall(TOKEN, top_k=5)
    assert len(calls) == 2, "latch did not clear after a foreign commit"


def test_recall_vec_latch_cleared_by_inprocess_reindex(store, monkeypatch):
    """A repair through reindex_vectors() clears the latch explicitly (the
    same connection's own commits do NOT bump its data_version, so the
    data_version probe alone could never observe an in-process repair)."""
    def boom(conn, embedding, k=20):
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_search", boom)
    store.recall(TOKEN, top_k=5)
    assert beam._RECALL_VEC_LATCH, "premise: the failing call latched"

    beam.reindex_vectors(store.conn, batch_size=8)
    assert beam._RECALL_VEC_LATCH == {}, "reindex must clear the latch"


def test_recall_non_sqlite_errors_still_propagate(store, monkeypatch):
    """Scope the catch to sqlite3.Error: a code bug from the embedding shape
    must surface loudly, not be silently converted into a degraded recall."""
    def boom(conn, embedding, k=20):
        raise TypeError("embedding shape bug")

    monkeypatch.setattr(beam, "_vec_search", boom)
    with pytest.raises(TypeError, match="embedding shape bug"):
        store.recall(TOKEN, top_k=5)


def test_consolidate_to_episodic_persists_row_and_logs_error_on_vec_failure(
    store, monkeypatch, caplog
):
    """The memory row is the payload; the vector is an index. A failing vec
    insert must leave the row readable, log at ERROR with table + rowid, and
    emit exactly one remediation hint."""
    def boom(conn, rowid, embedding, *, commit=True):
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_insert", boom)

    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        memory_id = store.consolidate_to_episodic(
            summary="consolidation survives a vec failure",
            source_wm_ids=["wm-1"],
            importance=0.7,
        )
        # A second failure must not repeat the hint.
        store.consolidate_to_episodic(
            summary="second consolidation survives too",
            source_wm_ids=["wm-2"],
            importance=0.7,
        )

    persisted = store.conn.execute(
        "SELECT content FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    assert persisted is not None, "the episodic row must survive the vec failure"
    assert persisted["content"] == "consolidation survives a vec failure"

    messages = [r.message for r in _error_records(caplog)]
    assert any("vec_episodes insert failed" in m for m in messages), messages
    assert any("rowid=" in m and "OperationalError" in m for m in messages), messages
    failures = [m for m in messages if "vec_episodes insert failed" in m]
    assert len(failures) == 2, "each failure is reported, not deduped"
    hints = [m for m in messages if "mnemosyne reindex --yes" in m]
    assert len(hints) == 1, f"reindex hint must be one-time: {hints}"


def test_refresh_episodic_embedding_degrades_on_vec_failure(store, monkeypatch, caplog):
    """Driven through its real caller (degrade_episodic): a sqlite3.Error from
    the refresh's vec write must not abort the content update. The row
    degrades to FTS recall and the failure is logged at ERROR.

    Pre-fix the call site was unguarded, so the OperationalError propagated
    into the caller's SAVEPOINT and rolled the content mutation back -- the
    row silently kept its old text while the caller counted no degradation."""
    from datetime import datetime, timedelta

    long_content = "QUOKKA_MIGRATION_DETAIL " * 30
    store.conn.execute(
        "UPDATE episodic_memory SET content = ?, tier = 2, created_at = ? WHERE id = 'em-1'",
        (
            long_content.strip(),
            (datetime.now() - timedelta(days=beam.TIER3_DAYS + 1)).isoformat(),
        ),
    )
    store.conn.commit()

    def boom(conn, insert_rowid, embedding, *, commit=True):
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_insert", boom)

    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        result = store.degrade_episodic(dry_run=False)

    assert result["tier2_to_tier3"] == 1, (
        f"the degradation must complete despite the vec failure: {result}"
    )
    persisted = store.conn.execute(
        "SELECT content, tier FROM episodic_memory WHERE id = 'em-1'"
    ).fetchone()
    assert persisted["tier"] == 3, "the tier transition must persist"
    assert persisted["content"] != long_content, (
        "the compressed content update must persist even when the vec write fails"
    )
    messages = [r.message for r in _error_records(caplog)]
    assert any("_refresh_episodic_embedding" in m for m in messages), messages
    assert any("mnemosyne reindex --yes" in m for m in messages), messages


def test_refresh_episodic_embedding_lets_non_sqlite_errors_propagate(
    store, monkeypatch
):
    """The refresh guard is scoped to sqlite3.Error so the caller's SAVEPOINT
    rollback keeps content/tier/vec atomic for code bugs (the contract pinned
    by test_degrade_vector.py::test_sqlite_vec_refresh_failure_rolls_back_row_and_vector)."""
    row = store.conn.execute(
        "SELECT rowid, id FROM episodic_memory WHERE id = 'em-1'"
    ).fetchone()

    def boom(conn, insert_rowid, embedding, *, commit=True):
        raise RuntimeError("simulated non-SQL refresh failure")

    monkeypatch.setattr(beam, "_vec_insert", boom)
    with pytest.raises(RuntimeError, match="simulated non-SQL"):
        store._refresh_episodic_embedding(row["id"], row["rowid"], "rewritten")


def test_import_from_dict_logs_error_on_vec_failure(store, monkeypatch, caplog):
    """Import failures matter: a dropped vector on import must be visible at
    ERROR with the table named, not a false INFO 'Regex extraction failed'."""
    def boom(conn, rowid, embedding, *, commit=True):
        raise sqlite3.OperationalError(BLOB_MISMATCH)

    monkeypatch.setattr(beam, "_vec_insert", boom)

    payload = {
        "episodic_memory": [
            {
                "rowid": 1,
                "id": "imported-1",
                "content": "imported episode",
                "source": "import",
                "timestamp": "2026-01-01T00:00:00",
                "importance": 0.5,
            }
        ],
        "episodic_embeddings": [{"rowid": 1, "embedding": [0.01] * 384}],
    }
    with caplog.at_level(logging.ERROR, logger="mnemosyne.core.beam"):
        store.import_from_dict(payload)

    messages = [r.message for r in _error_records(caplog)]
    assert any("import_from_dict" in m and "vec_episodes" in m for m in messages), messages
    assert not any("Regex extraction failed" in m for m in messages), messages
