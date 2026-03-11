"""Streamlit app for browsing PostgreSQL market tables.

Requirements implemented:
- Connect to local Docker PostgreSQL instance (defaults provided below).
- Discover all tables from `public` schema dynamically.
- Show first 5 rows for each table.
- Render each table inside an expandable block for удобство.
"""

import os
from typing import List

import pandas as pd
import psycopg2
from psycopg2 import sql
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Статистика рынка", page_icon="📈", layout="wide")
st.title("📈 Статистика рынка")
st.caption("Просмотр таблиц PostgreSQL (schema: public) и первых 5 строк каждой таблицы.")


# -----------------------------------------------------------------------------
# DB configuration
# Priority: environment variables -> requested default values
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "6f_@%DB&hA2+$f_"),
}


def get_connection() -> psycopg2.extensions.connection:
    """Create a PostgreSQL connection.

    Kept as a separate function so it can be reused by future modules and tests.
    """
    return psycopg2.connect(**DB_CONFIG)


def fetch_public_tables(conn: psycopg2.extensions.connection) -> List[str]:
    """Return all table names from public schema in a stable order."""
    tables_sql = """
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """
    tables_df = pd.read_sql_query(tables_sql, conn)
    return tables_df["tablename"].tolist()


def fetch_table_preview(
    conn: psycopg2.extensions.connection,
    table_name: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Load first `limit` rows for selected table safely.

    Uses psycopg2.sql.Identifier to avoid SQL injection in table names.
    """
    query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name))
    return pd.read_sql_query(query.as_string(conn), conn, params=[limit])


def render_table_section(conn: psycopg2.extensions.connection, table_name: str) -> None:
    """Render one table preview block with title + expander + dataframe."""
    st.markdown(f"### Таблица: `{table_name}`")
    with st.expander(f"Показать первые 5 строк: {table_name}", expanded=False):
        preview_df = fetch_table_preview(conn, table_name, limit=5)
        if preview_df.empty:
            st.info("Таблица существует, но не содержит строк.")
        else:
            st.dataframe(preview_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Main rendering logic
# -----------------------------------------------------------------------------
try:
    with get_connection() as conn:
        st.success(
            f"Подключение успешно: {DB_CONFIG['host']}:{DB_CONFIG['port']} / DB={DB_CONFIG['dbname']}"
        )

        table_names = fetch_public_tables(conn)
        st.subheader("Список таблиц схемы public")

        if not table_names:
            st.warning("В схеме public не найдено таблиц.")
        else:
            st.write(f"Найдено таблиц: **{len(table_names)}**")
            for table_name in table_names:
                render_table_section(conn, table_name)

except Exception as exc:
    st.error(f"Ошибка подключения или запроса к PostgreSQL: {exc}")
    st.info(
        "Проверьте параметры подключения и убедитесь, что контейнер `moex_postgres` "
        "запущен на 127.0.0.1:5433."
    )
