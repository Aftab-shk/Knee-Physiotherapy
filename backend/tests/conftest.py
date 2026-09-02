"""
Shared test setup.

The database URL is forced here, before any test module imports `db` — the
engine is created at import time, so setting it later has no effect. It is set
rather than defaulted on purpose: a developer with DATABASE_URL exported would
otherwise have the suite drop tables in whatever database that points at.
"""

import os
import sys
import tempfile
from pathlib import Path

_TMP_DB = Path(tempfile.mkdtemp(prefix="physio-tests-")) / "test.db"

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["JWT_SECRET"] = "test-only-secret-never-used-in-a-deployment"

# Argon2 runs at its real cost here (~85 ms a hash), which is most of the auth
# suite's runtime. Left alone deliberately: a test-only fast path through
# password hashing means the thing being tested is not the thing that ships.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """
    Build the schema once per run, through the app's own migrations.

    Doing this per test instead cost ~0.8 s each — the whole migration chain,
    including SQLite's batch-mode table rebuilds — which made the suite too slow
    to run often enough to be useful. Once per session keeps the migration path
    covered here, and test_migrations.py exercises it directly.
    """
    from sqlalchemy import text

    import db as _db
    import models as _models  # noqa: F401  (registers the mappings before drop_all)

    _db.Base.metadata.drop_all(bind=_db.engine)
    with _db.engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    _db.init_db()
    yield


def reset_database():
    """
    Empty every table, leaving the schema and its version stamp in place.

    Data only, on purpose. Tests must never create tables from the models: that
    would let a broken migration pass the whole suite, which is exactly what
    happened before — create_all() built the current columns, alembic_version
    survived the drop, and the next upgrade tried to add a column already there.
    """
    import db as _db
    import models as _models  # noqa: F401

    with _db.engine.begin() as conn:
        # Children before parents, so foreign keys are never briefly violated.
        for table in reversed(_db.Base.metadata.sorted_tables):
            conn.execute(table.delete())


def stub_inference(monkeypatch):
    """
    Replace KneeClassifier *before* the app starts.

    The lifespan builds a real one and assigns it to main.classifier, which
    means every test that enters TestClient was loading the 70 MB EfficientNet
    checkpoint from disk — about a second each, and by far the largest cost in
    the suite. Patching the class where the lifespan imports it from means the
    stub is what gets constructed, so the checkpoint is never touched.

    Tests still override main.classifier afterwards when they need particular
    predictions; this only removes the wasted load.
    """
    from test_api_security import StubClassifier

    try:
        import model.inference as _inference
    except ImportError:
        # No torch installed. The lifespan already handles that by serving
        # without a model, so there is nothing to stub.
        return

    monkeypatch.setattr(_inference, "KneeClassifier", StubClassifier)
