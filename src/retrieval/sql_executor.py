from typing import Any

from sqlalchemy import text

from src.core.db import engine


MAX_ROWS = 100

def validate_sql(sql: str) -> str:
    """
    Validate SQL before execution.

    Only read-only SELECT queries are allowed.
    """

    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    sql = sql.strip()

    if sql.upper() == "UNSUPPORTED":
        return "UNSUPPORTED"

    normalized = " ".join(sql.upper().split())

    if not normalized.startswith("SELECT "):
        raise ValueError(
            "Only SELECT queries are allowed."
        )

    if ";" in sql:
        raise ValueError(
            "Multiple SQL statements are not allowed."
        )

    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "EXECUTE",
        "CALL",
        "MERGE",
    ]

    for keyword in forbidden_keywords:
        if keyword in normalized:
            raise ValueError(
                f"Forbidden SQL operation detected: {keyword}"
            )

    forbidden_objects = [
        "PG_CATALOG",
        "INFORMATION_SCHEMA",
        "PG_CLASS",
        "PG_TABLES",
        "PG_ATTRIBUTE",
        "PG_DATABASE",
        "PG_USER",
    ]

    for object_name in forbidden_objects:
        if object_name in normalized:
            raise ValueError(
                "Access to PostgreSQL system catalogs is not allowed."
            )

    return sql


def _sanitize_value(value: Any) -> Any:
    """
    Convert PostgreSQL values into JSON-safe values.
    """

    if value is None:
        return None

    if hasattr(value, "as_tuple"):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass

    return value


def _sanitize_row(row: dict) -> dict:
    """
    Remove sensitive fields and make values JSON-safe.
    """

    sensitive_fields = {
        "password",
        "password_hash",
        "secret",
        "api_key",
        "token",
        "access_token",
        "refresh_token",
        "pin",
        "card_pin",
        "cvv",
    }

    result = {}

    for key, value in row.items():

        if key.lower() in sensitive_fields:
            continue

        result[key] = _sanitize_value(value)

    return result

async def execute_sql(
    sql: str,
    account_id: str | None = None,
) -> list[dict]:
    """
    Execute a validated read-only SQL query.

    If the generated SQL contains :account_id,
    the application must provide account_id.
    """

    validated_sql = validate_sql(sql)

    if validated_sql == "UNSUPPORTED":
        return []


    requires_account_id = ":account_id" in validated_sql

    if requires_account_id and not account_id:
        raise ValueError(
            "This SQL query requires an account_id."
        )

    parameters = {}

    if requires_account_id:
        parameters["account_id"] = account_id

    async with engine.connect() as connection:

        result = await connection.execute(
            text(validated_sql),
            parameters,
        )

        rows = result.mappings().fetchmany(
            MAX_ROWS
        )

        return [
            _sanitize_row(
                dict(row)
            )
            for row in rows
        ]

async def execute_sql_with_metadata(
    sql: str,
    account_id: str | None = None,
) -> dict:
    """
    Execute SQL and return structured execution information.
    """

    validated_sql = validate_sql(sql)

    if validated_sql == "UNSUPPORTED":

        return {
            "sql_query": None,
            "sql_result": [],
            "row_count": 0,
            "status": "unsupported",
        }

    rows = await execute_sql(
        validated_sql,
        account_id=account_id,
    )

    return {
        "sql_query": validated_sql,
        "sql_result": rows,
        "row_count": len(rows),
        "status": "success",
    }