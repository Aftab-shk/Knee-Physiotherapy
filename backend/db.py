"""
db.py — database engine, session lifecycle, and the ORM base.

The app was stateless until now: it graded an X-ray, printed a plan, and forgot.
Everything downstream of this file — progress history, clinician review, sharing
— needs a row somewhere, so this is where that starts.

Configuration
-------------
  DATABASE_URL   SQLAlchemy URL. Defaults to a SQLite file beside this package,
                 so `uvicorn main:app` works with nothing configured. Point it at
                 Postgres for anything real: postgresql+psycopg://user@host/db

Sessions are handed to endpoints through the `get_db` dependency, which always
closes them. Endpoints that touch the database are declared `def` rather than
`async def`, so FastAPI runs them in a threadpool — the driver here is blocking,
and calling it from the event loop would stall every other request.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("physio-backend.db")

# Resolved against this file rather than the working directory: `uvicorn` is run
# from backend/, pytest from the repo root, and a relative SQLite URL would
# otherwise create a second, empty database depending on where you started.
_DEFAULT_SQLITE = Path(__file__).resolve().parent / "physio.db"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_SQLITE}")

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    # SQLite otherwise refuses a connection created on one thread and used on
    # another, which is exactly what FastAPI's threadpool does. Safe here because
    # a Session is never shared between requests.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # Verify a pooled connection before handing it out. Without this, a database
    # that restarts (or a proxy that times connections out) hands back a dead
    # socket and the next request fails for no visible reason.
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "").lower() in ("1", "true", "yes"),
)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        """
        SQLite defaults are wrong for a server.

        foreign_keys is OFF by default, so a FK constraint is decorative until
        switched on per connection — a prescription could outlive the patient it
        belongs to. WAL lets reads proceed during a write, which matters as soon
        as more than one request is in flight.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


# Alembic emits `batch_op.create_foreign_key(None, ...)` for an unnamed
# constraint, and SQLite's batch mode — which rebuilds the table around the
# change — cannot rebuild a constraint it has no name for. That failed a
# migration once; a convention means every constraint from here on has a name
# without anyone having to remember to give it one.
NAMING_CONVENTION = {
    "ix":  "ix_%(table_name)s_%(column_0_N_name)s",
    "uq":  "uq_%(table_name)s_%(column_0_N_name)s",
    "ck":  "ck_%(table_name)s_%(constraint_name)s",
    "fk":  "fk_%(table_name)s_%(column_0_name)s",
    "pk":  "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for every model in models.py."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# The revision that describes the schema as it stood before Alembic existed.
# Databases created by the old create_all() carry those tables but no version
# stamp, so they are stamped here rather than being asked to create tables that
# are already there.
_BASELINE_REVISION = "1160bf4cd340"


def _alembic_config():
    from alembic.config import Config

    cfg = Config(str(Path(__file__).resolve().parent / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent / "migrations"))
    return cfg


def init_db() -> None:
    """
    Bring the database up to the current schema.

    Runs the same migrations in development as in production, so the two cannot
    drift. `create_all` is deliberately not used any more: it adds missing
    tables but never alters an existing one, which is silent data loss waiting
    to happen the first time a column changes.
    """
    from alembic import command
    from sqlalchemy import inspect

    from models import (  # noqa: F401  (registers the mappings)
        CareLink,
        Clinician,
        ExerciseSession,
        ExerciseSet,
        Patient,
        Prescription,
        PrescriptionAudit,
        ShareLink,
    )

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    cfg = _alembic_config()

    if tables and "alembic_version" not in tables:
        # A database from before migrations existed. Its tables match the
        # baseline, so record that and let the later revisions apply normally.
        logger.info("Adopting a pre-migration database — stamping %s", _BASELINE_REVISION)
        command.stamp(cfg, _BASELINE_REVISION)

    command.upgrade(cfg, "head")
    logger.info("Database ready at %s", _redact(DATABASE_URL))


def _redact(url: str) -> str:
    """Hide any password before a URL reaches the logs."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    creds, _, host = rest.rpartition("@")
    user = creds.split(":")[0]
    return f"{scheme}://{user}:***@{host}"
