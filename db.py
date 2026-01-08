from __future__ import annotations
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result
from config import get_database_url
from pre_execution_validation import clean_and_validate_sql, SQLValidationError
from langsmith import traceable


DATABASE_URL = get_database_url()
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10},
)

@traceable(run_type="tool", name="run_select")
def run_select(query: str) -> List[Dict[str, Any]]:
    """
    Validate query with shared guardrails, then execute.
    """
    try:
        safe_sql, _debug = clean_and_validate_sql(query)
    except SQLValidationError as e:
        # Surface a simple error to callers, but keep original reason in message
        raise ValueError(f"Invalid SQL blocked by guardrails: {e}") from e

    with engine.connect() as conn:
        result: Result = conn.execute(text(safe_sql))
        rows = [dict(r._mapping) for r in result]
    return rows
