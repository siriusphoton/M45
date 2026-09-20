from collections.abc import Iterator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.config import load_settings
from m45.database import create_database_engine


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    database_engine = create_database_engine(load_settings())

    try:
        yield database_engine
    finally:
        database_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with engine.connect() as connection:
        transaction = connection.begin()

        try:
            with Session(bind=connection) as database_session:
                yield database_session
        finally:
            transaction.rollback()
