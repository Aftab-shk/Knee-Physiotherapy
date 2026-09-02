"""
Migrations.

The rest of the suite builds its schema once, through init_db(), and then only
clears data — so these are the tests that actually exercise the migration chain,
including the two cases that are easy to get wrong and impossible to notice
until a deployment:

  * A fresh database gets every table and every column.
  * A database created before migrations existed is adopted rather than
    rebuilt. Those carry the baseline tables but no version stamp, so a naive
    `upgrade head` would try to CREATE tables that are already there.

The "old" database is built by running the migrations up to the baseline, not by
hand-editing today's models, so none of this needs touching each time a column
is added. Each test uses its own throwaway SQLite file and never touches the
database the rest of the suite shares.

Run:  python -m pytest backend/tests -q
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sqlalchemy", reason="migration tests need sqlalchemy")
pytest.importorskip("alembic", reason="migration tests need alembic")

BACKEND = Path(__file__).resolve().parents[1]

# The revision that predates every later change.
BASELINE = "1160bf4cd340"


def run_in_subprocess(body: str, db_path: Path) -> str:
    """
    Run a snippet against its own database, in its own process.

    A subprocess is the point: db.py builds its engine at import time from
    DATABASE_URL, so a second database cannot be reached from this one without
    reimporting the module from scratch.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        + textwrap.dedent(body)
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
        # Handled below, so a failure can carry the subprocess's own output.
        check=False,
        env={**os.environ, "DATABASE_URL": f"sqlite:///{db_path.as_posix()}"},
    )
    if result.returncode != 0:
        raise AssertionError("subprocess failed:\n" + result.stdout + "\n" + result.stderr)
    return result.stdout


def field(out: str, marker: str) -> str:
    """Pull one labelled line out of a subprocess's stdout."""
    return out.split(marker)[1].splitlines()[0].strip()


# Reports any column the models declare that the database does not have. Stated
# this way — rather than as a list of expected names — so a forgotten migration
# fails here instead of at runtime on someone's deployment.
REPORT_MISSING_COLUMNS = """
from sqlalchemy import inspect
_i = inspect(db.engine)
print("MISSING:", ",".join(sorted(
    t + "." + col.name
    for t, table in db.Base.metadata.tables.items()
    for col in table.columns
    if col.name not in {c["name"] for c in _i.get_columns(t)}
)) or "-")
"""

# The world before migrations: baseline tables, and no version stamp to say so.
# Anyone who ran this project early has exactly this on disk.
BUILD_LEGACY = f"""
import db
from alembic import command
from sqlalchemy import text
command.upgrade(db._alembic_config(), {BASELINE!r})
with db.engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
"""


def test_a_fresh_database_gets_the_whole_schema(tmp_path):
    out = run_in_subprocess(
        """
import db, models
db.init_db()
from sqlalchemy import inspect, text
print("TABLES:", ",".join(sorted(inspect(db.engine).get_table_names())))
with db.engine.connect() as c:
    print("HEAD:", c.execute(text("select version_num from alembic_version")).scalar())
"""
        + REPORT_MISSING_COLUMNS,
        tmp_path / "fresh.db",
    )

    tables = set(field(out, "TABLES:").split(","))
    assert {"patients", "prescriptions", "exercise_sessions", "exercise_sets"} <= tables
    assert "alembic_version" in tables
    assert field(out, "HEAD:"), "no revision recorded"
    assert field(out, "MISSING:") == "-"


def test_a_database_from_before_migrations_is_adopted(tmp_path):
    """
    The upgrade path for anyone who ran this project before Alembic existed.
    init_db has to recognise an unstamped database and stamp it, rather than
    asking Alembic to create tables that are already there.
    """
    db_path = tmp_path / "legacy.db"

    before = run_in_subprocess(
        BUILD_LEGACY + """
from sqlalchemy import inspect
print("STAMPED:", "alembic_version" in inspect(db.engine).get_table_names())
""",
        db_path,
    )
    assert field(before, "STAMPED:") == "False"

    after = run_in_subprocess(
        "import db, models\ndb.init_db()\n" + REPORT_MISSING_COLUMNS,
        db_path,
    )
    missing = field(after, "MISSING:")
    assert missing == "-", f"the legacy database was not brought up to date: {missing}"


def test_existing_rows_survive_the_upgrade(tmp_path):
    """
    SQLite cannot ALTER a column in place, so env.py runs these in batch mode:
    the table is rebuilt around the change and the data copied across. That is
    the step that loses patient records if it is configured wrongly.

    Seeded with raw SQL rather than the ORM, because the models describe today's
    schema and this row has to be written into yesterday's.
    """
    db_path = tmp_path / "withdata.db"

    run_in_subprocess(
        BUILD_LEGACY + """
with db.engine.begin() as conn:
    conn.execute(text(
        "INSERT INTO patients (id, email, password_hash, created_at) "
        "VALUES ('pat1', 'ada@example.com', 'x', '2026-01-01 00:00:00')"))
    conn.execute(text(
        "INSERT INTO exercise_sessions "
        "(id, patient_id, client_session_id, exercise_name, tracked_joint, knee_side, "
        " angle_limit, target_sets, started_at, last_set_at, sets_completed, completed, view_verified) "
        "VALUES ('ses1', 'pat1', 'legacy-sitting', 'Mini Squat', 'knee', 'left', "
        "        60, 3, '2026-01-02 09:00:00', '2026-01-02 09:10:00', 2, 0, 1)"))
    conn.execute(text(
        "INSERT INTO exercise_sets "
        "(id, session_id, set_index, reps_completed, duration_seconds, peak_flexion_deg, "
        " breach_count, breach_seconds, suspended_seconds, recorded_at) "
        "VALUES ('set1', 'ses1', 1, 10, 40.0, 51.5, 0, 0.0, 0.0, '2026-01-02 09:05:00')"))
print("SEEDED")
""",
        db_path,
    )

    out = run_in_subprocess(
        """
import db, models
db.init_db()
from sqlalchemy.orm import Session
with Session(db.engine) as s:
    sess = s.query(models.ExerciseSession).one()
    st = s.query(models.ExerciseSet).one()
    print("EXERCISE:", sess.exercise_name)
    print("SETS_COMPLETED:", sess.sets_completed)
    print("PEAK:", st.peak_flexion_deg)
    print("PAIN:", sess.pain_before)
    print("SURGERY:", s.query(models.Patient).one().surgery_date)
    print("PATIENTS:", s.query(models.Patient).count())
""",
        db_path,
    )

    assert field(out, "EXERCISE:") == "Mini Squat"
    assert field(out, "SETS_COMPLETED:") == "2"
    assert field(out, "PEAK:") == "51.5"
    assert field(out, "PATIENTS:") == "1"
    # Columns added after this row was written exist, and are empty for it.
    assert field(out, "PAIN:") == "None"
    assert field(out, "SURGERY:") == "None"


def test_running_init_db_twice_changes_nothing(tmp_path):
    """
    A restarting app runs init_db again. It must be a no-op, not a second
    attempt at the migrations it already applied.
    """
    out = run_in_subprocess(
        """
import db, models
db.init_db()
db.init_db()
db.init_db()
from sqlalchemy import inspect
# Counted against the models rather than a literal, so adding a table does not
# turn this into a failing test that says nothing.
print("EXPECTED:", len(db.Base.metadata.tables) + 1)   # + alembic_version
print("TABLES:", len(inspect(db.engine).get_table_names()))
""",
        tmp_path / "twice.db",
    )
    assert field(out, "TABLES:") == field(out, "EXPECTED:")


def test_every_revision_is_reachable_from_a_single_head():
    """
    Two heads mean someone branched the history, and `upgrade head` then fails
    with an error nobody reads until a deploy. Cheap to check, expensive to
    discover late.
    """
    from alembic.script import ScriptDirectory

    sys.path.insert(0, str(BACKEND))
    import db

    script = ScriptDirectory.from_config(db._alembic_config())
    heads = script.get_heads()
    assert len(heads) == 1, f"expected one head, found {heads}"
    assert len(list(script.walk_revisions())) >= 3
