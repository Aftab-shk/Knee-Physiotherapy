"""
Alembic environment.

The URL comes from db.py rather than alembic.ini, so `alembic upgrade` can never
run against a different database than the application does.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: F401  (imported for its side effect: registering the mappings)
from db import DATABASE_URL, Base, engine

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False, and it matters a great deal.
    #
    # The default is True, which switches off every logger created before this
    # line. init_db() runs the migrations during application startup, so the
    # default silenced the entire app for the life of the process: no
    # "Registered patient", no prescription audit line, no OOD rejection, no
    # rate-limit warning, and no password-reset link in development — all of it
    # gone the moment the server finished migrating, with nothing to indicate
    # logging had stopped.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER a column in place. Batch mode rebuilds the
            # table around the change, which is the only way most alterations
            # work there at all.
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
