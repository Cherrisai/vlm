"""SQLite connection management and low-level data access layer."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import SETTINGS
from database.schema import ALL_SCHEMA_STATEMENTS
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteManager:
    """Thin wrapper around sqlite3 that manages connections and schema setup."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or SETTINGS.database.db_path
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            logger.exception("Database operation failed, transaction rolled back.")
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self.get_connection() as connection:
            cursor = connection.cursor()
            for statement in ALL_SCHEMA_STATEMENTS:
                cursor.execute(statement)
        logger.info("Database schema initialized at %s", self.db_path)

    def execute(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE statement and return the last row id."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid

    def fetch_all(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a SELECT statement and return all rows."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def fetch_one(self, query: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute a SELECT statement and return a single row."""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()


_DB_INSTANCE: SQLiteManager | None = None


def get_db() -> SQLiteManager:
    """Return a process-wide singleton SQLiteManager instance."""
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = SQLiteManager()
    return _DB_INSTANCE
