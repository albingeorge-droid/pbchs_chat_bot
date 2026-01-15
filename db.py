from __future__ import annotations
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Result
from config import get_database_url
from pre_execution_validation import clean_and_validate_sql, SQLValidationError
import re
from langsmith import traceable


DATABASE_URL = get_database_url()
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 10},
)

def _remove_limit_clause(sql: str) -> str:
    """
    Remove LIMIT clause from SQL query if present.
    
    Handles cases like:
    - LIMIT 100;
    - LIMIT 50
    - limit 25;
    
    Returns the SQL without LIMIT clause.
    """
    # Remove LIMIT clause (case-insensitive)
    # Pattern matches: LIMIT followed by optional whitespace, digits, optional whitespace, optional semicolon
    cleaned = re.sub(
        r'\s*LIMIT\s+\d+\s*;?\s*$',
        '',
        sql,
        flags=re.IGNORECASE
    )
    
    # Ensure the query still ends with a semicolon
    cleaned = cleaned.rstrip()
    if not cleaned.endswith(';'):
        cleaned += ';'
    
    return cleaned


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

    # NEW: Remove LIMIT clause before execution
    final_sql = _remove_limit_clause(safe_sql)

    with engine.connect() as conn:
        result: Result = conn.execute(text(final_sql))
        rows = [dict(r._mapping) for r in result]
    return rows
