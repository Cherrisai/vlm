"""SQL schema definitions for Vision Intelligence Studio."""

from __future__ import annotations

CREATE_PROMPT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS prompt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,
    image_name TEXT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    inference_time REAL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
"""

CREATE_REQUEST_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_type TEXT NOT NULL,
    task_type TEXT NOT NULL,
    similarity_score REAL,
    inference_time REAL DEFAULT 0.0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

CREATE_INDEX_PROMPT_HISTORY_CREATED_AT = """
CREATE INDEX IF NOT EXISTS idx_prompt_history_created_at
ON prompt_history (created_at);
"""

CREATE_INDEX_REQUEST_LOG_CREATED_AT = """
CREATE INDEX IF NOT EXISTS idx_request_log_created_at
ON request_log (created_at);
"""

ALL_SCHEMA_STATEMENTS = [
    CREATE_PROMPT_HISTORY_TABLE,
    CREATE_REQUEST_LOG_TABLE,
    CREATE_INDEX_PROMPT_HISTORY_CREATED_AT,
    CREATE_INDEX_REQUEST_LOG_CREATED_AT,
]
