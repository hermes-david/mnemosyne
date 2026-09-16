"""Pure runtime diagnostics shared by doctor and legacy diagnose.

This module intentionally has no logging, database, provider, repair, or CLI
dependencies.  Keeping it neutral lets ``mnemosyne.doctor`` report runtime
capabilities without importing the mutable ``mnemosyne.diagnose`` command.
"""

import importlib.metadata
import platform
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


# --- Vector-index integrity (on-demand, doctor level) -----------------------
#
# A torn vec0 write (chunk metadata committed, vector blob left at zero
# length) makes every KNN query on the table raise ``vectors blob size
# doesn't match``, which kills dense recall until the index is rebuilt. The
# KNN itself validates blob size at zero marginal cost, so no per-query
# pre-flight is warranted on the hot path; a standing detector belongs here,
# where the operator pays the cost once per doctor run.
#
# vec0 keeps its vectors in shadow tables named ``<table>_vector_chunksNN``
# as fixed-capacity blobs: 1024 rows x dim x bytes-per-element
# (int8 -> 1, bit -> dim/8, float32 -> 4). Shadow tables are ordinary
# tables, so reading them needs no extension loading -- which is why this
# check still works in a runtime where vec0 cannot be loaded.
_VEC0_DDL = re.compile(r"\bUSING\s+vec0\b", re.IGNORECASE)
_VEC0_DIM = re.compile(r"\[(\d+)\]")
_VEC0_CHUNK_CAPACITY = 1024
# Bounded so a pathological store cannot turn a doctor run into a huge scan;
# 1000 chunks covers 1M+ rows per table (each chunk holds 1024 rows).
_VEC_INDEX_MAX_CHUNKS = 1000
_VEC_INDEX_TABLES = ("vec_episodes", "vec_working", "vec_facts")
_VEC_INDEX_REMEDY = (
    "Run 'mnemosyne reindex --yes' (with providers stopped) to rebuild "
    "the vector index."
)


def _quote_sql_identifier(identifier: str) -> str:
    """Double-quote a SQLite identifier read from trusted schema metadata."""

    return '"' + identifier.replace('"', '""') + '"'


def _vec_bytes_per_row(declared_type: str, dim: int) -> Optional[float]:
    """Bytes stored per vector row in a vec0 chunk, or None if unrecognized.

    Verified against sqlite-vec 0.1.9 (``length(vectors)`` on a one-row
    table): int8[8] -> 8, bit[8] -> 1, float32[8] -> 32. A chunk holds 1024
    rows, so the expected blob size is ``1024 * bytes_per_row`` -- for the
    live int8[1024] store that is 1024 * 1024 = 1048576, the size the
    incident's error message expected.
    """

    if "int8" in declared_type:
        return float(dim)
    if "bit" in declared_type:
        if dim % 8:
            return None  # bit vectors pack into whole bytes
        return dim / 8
    return float(dim * 4)  # float32 / float


def vec_index_integrity(conn: sqlite3.Connection) -> Tuple[str, str]:
    """Compare each vec0 table's chunk blobs against the size its DDL implies.

    Returns ``(status, detail)`` where status is:

    - ``OK`` -- every existing chunk blob carries its full expected capacity;
    - ``CORRUPT`` -- at least one chunk blob is short (the torn-write shape),
      with the table/chunk named and the reindex remedy prescribed;
    - ``NO`` -- the store has no vec0 tables to check; or
    - ``ERROR`` -- the schema metadata itself could not be read.

    The read is bounded and metadata-only: chunk blob sizes come from
    ``length(vectors)``, not from the blob contents.
    """

    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    except sqlite3.Error:
        return "ERROR", "vector-index metadata could not be read"

    ddls = {str(row[0]): str(row[1] or "") for row in rows}
    checked: list[str] = []
    truncated = False
    problems: list[str] = []

    for table in _VEC_INDEX_TABLES:
        ddl = ddls.get(table)
        if not ddl or not _VEC0_DDL.search(ddl):
            continue
        dim_match = _VEC0_DIM.search(ddl)
        if not dim_match:
            continue
        dim = int(dim_match.group(1))
        bytes_per_row = _vec_bytes_per_row(ddl, dim)
        if bytes_per_row is None:
            problems.append(f"{table}: unrecognized vec0 encoding in its DDL")
            continue
        expected = int(_VEC0_CHUNK_CAPACITY * bytes_per_row)
        checked.append(table)
        shadows = sorted(
            name for name in ddls if name.startswith(f"{table}_vector_chunks")
        )
        if not shadows:
            # No shadow chunk table at all: the index was never materialized
            # (an empty table keeps one) or the schema is broken.
            problems.append(f"{table}: no vector chunk shadow table present")
            continue
        for shadow in shadows:
            try:
                blob_rows = conn.execute(
                    f"SELECT rowid, length(vectors) FROM {_quote_sql_identifier(shadow)} "
                    f"LIMIT {_VEC_INDEX_MAX_CHUNKS + 1}"
                ).fetchall()
            except sqlite3.Error:
                problems.append(f"{shadow}: vector chunk blob is unreadable")
                continue
            if len(blob_rows) > _VEC_INDEX_MAX_CHUNKS:
                truncated = True
                blob_rows = blob_rows[:_VEC_INDEX_MAX_CHUNKS]
            for chunk_rowid, blob_length in blob_rows:
                if blob_length != expected:
                    # Phrasing matters: adjacent large bare numbers (e.g.
                    # "1048576 (1024-dim") are matched by the doctor's phone
                    # redactor and the expected size -- the single most
                    # diagnostic field -- is replaced with <redacted-phone>.
                    # Keep units between the numbers so both survive.
                    problems.append(
                        f"{shadow} rowid={chunk_rowid} holds {blob_length} bytes; "
                        f"expected {expected} bytes for its {dim}-dim vec0 chunk"
                    )

    if problems:
        detail = "; ".join(problems[:5])
        if len(problems) > 5:
            detail += f"; (+{len(problems) - 5} more)"
        return "CORRUPT", f"{detail}. {_VEC_INDEX_REMEDY}"
    if not checked:
        return "NO", "no vec0 vector tables found"
    detail = f"chunk blobs match declared dimensions for {', '.join(checked)}"
    if truncated:
        detail += f" (first {_VEC_INDEX_MAX_CHUNKS} chunks per table checked)"
    return "OK", detail


def collect_runtime_diagnostics(conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
    """Run pure runtime, dependency, and capability checks without a provider.

    ``conn`` is optional. When a read-only database connection is supplied,
    the vector-index integrity probe runs against it; without one, that check
    is skipped (the caller has no store to inspect).
    """

    checks: list[dict[str, str]] = []

    def add(category: str, check: str, status: str, detail: str = "") -> None:
        checks.append({"category": category, "check": check, "status": status, "detail": detail})


    add("env", "python_version", "OK", sys.version.split()[0])
    add("env", "platform", "OK", platform.platform())
    # Report only the executable name: an absolute interpreter path can reveal
    # a user's home directory or virtual-environment layout in diagnostics.
    add("env", "python_executable", "OK", Path(sys.executable).name)

    try:
        import mnemosyne

        version = getattr(mnemosyne, "__version__", None)
        if not version:
            version = importlib.metadata.version("mnemosyne-memory")
        add("package", "mnemosyne_version", "OK", str(version))
    except Exception:
        add("package", "mnemosyne_version", "ERROR", "package version unavailable")

    for name, module in {
        "fastembed": "fastembed",
        "sqlite_vec": "sqlite_vec",
        "numpy": "numpy",
        "huggingface_hub": "huggingface_hub",
    }.items():
        try:
            dependency = __import__(module)
            add("deps", name, "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
        except ImportError:
            add("deps", name, "MISSING")
        except Exception:
            add("deps", name, "ERROR", "dependency import failed")

    try:
        dependency = __import__("ctransformers")
        add("deps", "ctransformers", "OK", f"version={getattr(dependency, '__version__', 'unknown')}")
    except ImportError:
        add("deps", "ctransformers", "OPTIONAL", "optional local-GGUF fallback dependency not installed")
    except Exception:
        add("deps", "ctransformers", "ERROR", "dependency import failed")

    try:
        from mnemosyne.core import embeddings as _embeddings

        add("core", "embeddings_available", "YES" if _embeddings.available() else "NO")
        add("core", "embeddings_model", "OK", _embeddings._DEFAULT_MODEL)
        # Surface the resolved dimension so operators can confirm their
        # MNEMOSYNE_EMBEDDING_DIM / model-table resolution via the doctor,
        # complementing the fail-loud unknown-model resolver.
        add("core", "embeddings_dim", "OK", str(_embeddings.EMBEDDING_DIM))
    except Exception:
        add("core", "embeddings_available", "ERROR", "embeddings capability unavailable")

    try:
        from mnemosyne.core.beam import _SQLITE_VEC_AVAILABLE

        vec_can_load = False
        if _SQLITE_VEC_AVAILABLE:
            try:
                import sqlite_vec

                test_conn = sqlite3.connect(":memory:")
                try:
                    test_conn.enable_load_extension(True)
                    sqlite_vec.load(test_conn)
                    vec_can_load = True
                finally:
                    test_conn.close()
            except Exception:
                vec_can_load = False
        add("core", "sqlite_vec_available", "YES" if vec_can_load else "NO")
        if _SQLITE_VEC_AVAILABLE and not vec_can_load:
            add("core", "sqlite_vec_warning", "NO", "extension loading unavailable")
    except Exception:
        add("core", "sqlite_vec", "ERROR", "sqlite-vec capability unavailable")

    if conn is not None:
        # On-demand, offline integrity probe: a torn vec0 chunk bricks dense
        # recall with a per-query error, so the standing detector lives here
        # rather than on the hot path (see vec_index_integrity).
        try:
            status, detail = vec_index_integrity(conn)
        except Exception:
            status, detail = "ERROR", "vector-index integrity probe failed"
        add("index", "vec_index_integrity", status, detail)

    statuses = {entry["status"] for entry in checks}
    overall = (
        "unavailable"
        if statuses & {"ERROR", "CORRUPT"}
        else "warning"
        if statuses & {"MISSING", "NO"}
        else "ok"
    )
    return {"status": overall, "checks": checks}