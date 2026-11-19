from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from trading_system.storage.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    # Without this, fileConfig() switches off every logger that already exists. Run in a process
    # that has imported the application (the migration tests do), that silenced its logs.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


def _database_url() -> str:
    """`alembic -x db_url=<url> upgrade head` builds another database than the one in alembic.ini
    (the third paper account's var/paper3.db, docs/paper-third-account.md). Without -x the url of
    alembic.ini is used, exactly as before."""
    url = context.get_x_argument(as_dictionary=True).get("db_url") or config.get_main_option(
        "sqlalchemy.url"
    )
    if not url:
        raise RuntimeError(
            "No database url: set sqlalchemy.url in alembic.ini or pass -x db_url=..."
        )
    return url


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = dict(config.get_section(config.config_ini_section, {}))
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
