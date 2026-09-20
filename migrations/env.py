from logging.config import fileConfig

from alembic import context

from m45.config import load_settings
from m45.database import create_database_engine
from m45.source_history import SourceMessage

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SourceMessage.metadata


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""
    context.configure(
        url=str(load_settings().database_url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations through the configured application database."""
    engine = create_database_engine(load_settings())

    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
