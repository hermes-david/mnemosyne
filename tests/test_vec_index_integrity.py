"""Tests for the doctor-level vec_index_integrity check.

A torn vec0 write (chunk metadata committed, vector blob left at zero length)
bricks every KNN on that table with ``vectors blob size doesn't match``, and
pre-hardening it also killed recall's FTS branch. The standing detector is
``mnemosyne.runtime_diagnostics.vec_index_integrity``: it compares each vec0
table's shadow chunk blobs against the size the table's own DDL implies
(1024-row chunk capacity x dim x bytes-per-element).

The checker takes a ``conn``, so it is exercised here against a real
sqlite-vec store (healthy + torn) -- shadow tables are ordinary tables and
tolerate direct mutation, which is how the torn fixture is built.
"""
from __future__ import annotations

import sqlite3

import pytest

from mnemosyne import doctor, runtime_diagnostics

DIM = 8


@pytest.fixture
def vec_store(tmp_path):
    """A real sqlite-vec store with one row in vec_episodes and vec_working."""
    sqlite_vec = pytest.importorskip("sqlite_vec")

    db = tmp_path / "vec_integrity.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[{DIM}])")
    conn.execute(f"CREATE VIRTUAL TABLE vec_working USING vec0(embedding int8[{DIM}])")
    payload = "[" + ",".join(["0.25"] * DIM) + "]"
    for table in ("vec_episodes", "vec_working"):
        conn.execute(
            f"INSERT INTO {table}(rowid, embedding) "
            f"VALUES (1, vec_quantize_int8(?, 'unit'))",
            (payload,),
        )
    conn.commit()
    yield conn
    conn.close()


def _tear(conn, table):
    """Zero a table's first vector chunk blob, exactly the incident's shape."""
    conn.execute(f"UPDATE {table}_vector_chunks00 SET vectors = X'' WHERE rowid = 1")
    conn.commit()


def test_vec_index_integrity_ok_on_healthy_store(vec_store):
    """Healthy chunk blobs must report OK, with no false positives."""
    status, detail = runtime_diagnostics.vec_index_integrity(vec_store)
    assert status == "OK", detail
    assert "vec_episodes" in detail and "vec_working" in detail, detail


def test_vec_index_integrity_flags_torn_chunk(vec_store):
    """A zero-length chunk blob must be CORRUPT, naming the shadow table, and
    prescribe the reindex remedy."""
    _tear(vec_store, "vec_episodes")

    status, detail = runtime_diagnostics.vec_index_integrity(vec_store)

    assert status == "CORRUPT", detail
    assert "vec_episodes_vector_chunks00" in detail, detail
    assert "mnemosyne reindex" in detail, detail
    # The healthy table must not be implicated.
    assert "vec_working_vector_chunks00" not in detail, detail


def test_vec_index_integrity_reports_no_vec_tables(tmp_path):
    """A store without vec0 tables is NO, not OK and not CORRUPT."""
    conn = sqlite3.connect(str(tmp_path / "plain.db"))
    conn.execute("CREATE TABLE working_memory (id TEXT PRIMARY KEY)")
    conn.commit()
    try:
        status, detail = runtime_diagnostics.vec_index_integrity(conn)
    finally:
        conn.close()
    assert status == "NO", detail


def test_vec_index_integrity_reads_do_not_need_extension_loading(tmp_path):
    """The probe must work on a read-only connection with no vec0 loaded:
    it reads shadow tables and DDL, never KNN-queries the virtual table."""
    import sqlite_vec

    db = tmp_path / "ro.db"
    writable = sqlite3.connect(str(db))
    writable.enable_load_extension(True)
    sqlite_vec.load(writable)
    writable.enable_load_extension(False)
    writable.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding bit[{DIM}])")
    writable.execute(
        "INSERT INTO vec_episodes(rowid, embedding) VALUES (1, vec_quantize_binary(?))",
        ("[" + ",".join(["1.0"] * DIM) + "]",),
    )
    writable.commit()
    _tear(writable, "vec_episodes")
    writable.close()

    readonly = doctor.open_readonly_doctor_db(db)
    try:
        status, detail = runtime_diagnostics.vec_index_integrity(readonly)
    finally:
        readonly.close()
    assert status == "CORRUPT", detail
    assert "vec_episodes_vector_chunks00" in detail, detail


def test_vec_index_integrity_detail_survives_doctor_redaction(vec_store):
    """The expected byte count is the single most diagnostic field; the
    doctor's phone/PII redactor must not eat it. Adjacent bare large numbers
    (``1048576 (1024-`` in the production store) tripped the phone matcher and
    replaced the expected size with <redacted-phone>."""
    _tear(vec_store, "vec_episodes")

    _status, detail = runtime_diagnostics.vec_index_integrity(vec_store)
    redacted = doctor.safe_preview(detail, max_length=240)

    assert "redacted-phone" not in redacted, redacted
    # Both numbers in the comparison must survive: the observed size and the
    # expected one (8-dim int8 chunk -> 1024 * 8 = 8192 bytes).
    assert "8192" in redacted, redacted
    assert f"{DIM}-dim" in redacted, redacted
    assert "vec_episodes_vector_chunks00" in redacted, redacted
    assert "redacted-path" not in redacted, redacted


def test_vec_index_integrity_production_dimensions_survive_redaction(tmp_path):
    """The incident's own numbers: a torn int8[1024] chunk reports the
    1048576-byte expectation and the report boundary keeps it readable."""
    import sqlite_vec

    db = tmp_path / "prod_dims.db"
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[1024])")
    conn.execute(
        "INSERT INTO vec_episodes(rowid, embedding) VALUES (1, vec_quantize_int8(?, 'unit'))",
        ("[" + ",".join(["0.01"] * 1024) + "]",),
    )
    conn.commit()
    assert conn.execute(
        "SELECT length(vectors) FROM vec_episodes_vector_chunks00"
    ).fetchone()[0] == 1048576, "premise: the healthy int8[1024] chunk is 1 MiB"
    _tear(conn, "vec_episodes")
    try:
        status, detail = runtime_diagnostics.vec_index_integrity(conn)
    finally:
        conn.close()

    assert status == "CORRUPT"
    assert "1048576" in detail, detail
    assert "1024" in detail, detail
    redacted = doctor.safe_preview(detail, max_length=240)
    assert "redacted-phone" not in redacted, redacted
    assert "1048576" in redacted, redacted  # the diagnostic survives the boundary


def test_runtime_diagnostics_skips_integrity_without_a_connection(monkeypatch):
    """No connection supplied -> no index check emitted (the adapter runs
    before the doctor opens its store)."""
    result = runtime_diagnostics.collect_runtime_diagnostics()
    assert not [c for c in result["checks"] if c["check"] == "vec_index_integrity"]


def test_runtime_diagnostics_includes_integrity_with_a_connection(vec_store, monkeypatch):
    """With a connection, the check is emitted with category 'index' and the
    sanitization allowlist keeps it in the rendered report."""
    _tear(vec_store, "vec_episodes")

    result = runtime_diagnostics.collect_runtime_diagnostics(vec_store)
    entries = {c["check"]: c for c in result["checks"]}
    assert entries["vec_index_integrity"]["status"] == "CORRUPT", entries
    assert entries["vec_index_integrity"]["category"] == "index"
    assert result["status"] == "unavailable", "corruption must not read as ok"

    sanitized = doctor._sanitize_runtime_diagnostics(result)
    kept = [c for c in sanitized["checks"] if c["check"] == "vec_index_integrity"]
    assert kept, "vec_index_integrity must survive the report allowlist"
    assert kept[0]["status"] == "CORRUPT"


def test_doctor_report_flags_torn_index_on_a_real_store(tmp_path):
    """End-to-end through build_doctor_report: a torn chunk surfaces as a
    critical finding AND a CORRUPT runtime check, on a read-only connection."""
    import sqlite_vec

    from mnemosyne.doctor import (
        SEVERITY_CRITICAL,
        build_doctor_report,
        doctor_report_payload,
        render_doctor_markdown,
    )

    db = tmp_path / "doctor_torn.db"
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[{DIM}])")
    conn.execute(
        "INSERT INTO vec_episodes(rowid, embedding) VALUES (1, vec_quantize_int8(?, 'unit'))",
        ("[" + ",".join(["0.25"] * DIM) + "]",),
    )
    conn.commit()
    _tear(conn, "vec_episodes")
    conn.close()

    report = build_doctor_report("default", db)

    checks = {c["check"]: c for c in report.runtime_diagnostics["checks"]}
    assert checks["vec_index_integrity"]["status"] == "CORRUPT", checks
    assert report.runtime_diagnostics["status"] == "unavailable"
    criticals = [f for f in report.findings if f.severity == SEVERITY_CRITICAL]
    assert any(f.code == "vec_index_integrity" for f in criticals), report.findings
    # The remedy must reach the rendered artifact an operator reads.
    markdown = render_doctor_markdown(doctor_report_payload(report))
    assert "vec_index_integrity" in markdown
    assert "mnemosyne reindex" in markdown


def test_doctor_report_reports_ok_index_on_a_healthy_store(tmp_path):
    """The healthy case must not produce a finding or an unavailable status."""
    import sqlite_vec

    from mnemosyne.doctor import build_doctor_report

    db = tmp_path / "doctor_ok.db"
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding int8[{DIM}])")
    conn.execute(
        "INSERT INTO vec_episodes(rowid, embedding) VALUES (1, vec_quantize_int8(?, 'unit'))",
        ("[" + ",".join(["0.25"] * DIM) + "]",),
    )
    conn.commit()
    conn.close()

    report = build_doctor_report("default", db)

    checks = {c["check"]: c for c in report.runtime_diagnostics["checks"]}
    assert checks["vec_index_integrity"]["status"] == "OK", checks
    assert not [f for f in report.findings if f.code == "vec_index_integrity"]
