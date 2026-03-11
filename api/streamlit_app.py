"""High-performance Streamlit dashboard for bond emitters from PostgreSQL.

Design goals:
- Read only from `bond_emitters_market_view` in Streamlit.
- Keep queries lightweight by using server-side filtering/sorting/LIMIT.
- Keep code modular so new bond-market tables/queries can be added easily.
"""

import os
from typing import Dict, List, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Эмитенты облигаций", page_icon="📈", layout="wide")
st.title("📈 Эмитенты облигаций")
st.caption("Источник данных: PostgreSQL view `public.bond_emitters_market_view`.")

# -----------------------------------------------------------------------------
# DB settings (env first, then required defaults)
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "6f_@%DB&hA2+$f_"),
}

DEFAULT_LIMIT = 1000
VIEW_NAME = "public.bond_emitters_market_view"

# Whitelist for safe ORDER BY.
SORT_OPTIONS: Dict[str, Tuple[str, str]] = {
    "bonds_count ↓": ("bonds_count", "DESC"),
    "bonds_count ↑": ("bonds_count", "ASC"),
    "last_maturity_date ↓": ("last_maturity_date", "DESC"),
    "last_maturity_date ↑": ("last_maturity_date", "ASC"),
}


def get_connection() -> psycopg2.extensions.connection:
    """Open PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def list_public_relations(conn: psycopg2.extensions.connection) -> List[str]:
    """Return list of public schema tables and views."""
    query = """
        SELECT table_name AS relation_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        UNION
        SELECT table_name AS relation_name
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY relation_name;
    """
    rel_df = pd.read_sql_query(query, conn)
    return rel_df["relation_name"].tolist()


def fetch_emitters(
    conn: psycopg2.extensions.connection,
    emitter_filter: str,
    sort_option: str,
    limit: int,
) -> pd.DataFrame:
    """Fetch emitters from market view with server-side filter/sort/limit.

    This keeps app fast on large datasets by avoiding full-table fetches.
    """
    sort_col, sort_dir = SORT_OPTIONS[sort_option]

    # Build WHERE fragment only when filter is provided (lazy-loading style).
    where_sql = sql.SQL("")
    params: List[object] = []
    if emitter_filter.strip():
        where_sql = sql.SQL("WHERE emitent_title ILIKE %s")
        params.append(f"%{emitter_filter.strip()}%")

    query = sql.SQL(
        """
        SELECT
            emitent_title,
            emitent_inn,
            bonds_count,
            first_issue_date,
            last_maturity_date
        FROM {view}
        {where_clause}
        ORDER BY {sort_col} {sort_dir}, emitent_title ASC
        LIMIT %s
        """
    ).format(
        view=sql.SQL(VIEW_NAME),
        where_clause=where_sql,
        sort_col=sql.Identifier(sort_col),
        sort_dir=sql.SQL(sort_dir),
    )
    params.append(limit)

    df = pd.read_sql_query(query.as_string(conn), conn, params=params)

    # Pandas normalization for consistent UI behavior.
    if not df.empty:
        df["bonds_count"] = pd.to_numeric(df["bonds_count"], errors="coerce").fillna(0).astype(int)
        df["first_issue_date"] = pd.to_datetime(df["first_issue_date"], errors="coerce").dt.date
        df["last_maturity_date"] = pd.to_datetime(df["last_maturity_date"], errors="coerce").dt.date

    return df


def fetch_top10(conn: psycopg2.extensions.connection, order_column: str) -> pd.DataFrame:
    """Fetch top-10 rows from market view for a specific metric."""
    query = sql.SQL(
        """
        SELECT
            emitent_title,
            emitent_inn,
            bonds_count,
            first_issue_date,
            last_maturity_date
        FROM {view}
        ORDER BY {order_column} DESC NULLS LAST, emitent_title ASC
        LIMIT 10
        """
    ).format(view=sql.SQL(VIEW_NAME), order_column=sql.Identifier(order_column))

    df = pd.read_sql_query(query.as_string(conn), conn)
    if not df.empty:
        df["bonds_count"] = pd.to_numeric(df["bonds_count"], errors="coerce").fillna(0).astype(int)
        df["first_issue_date"] = pd.to_datetime(df["first_issue_date"], errors="coerce").dt.date
        df["last_maturity_date"] = pd.to_datetime(df["last_maturity_date"], errors="coerce").dt.date
    return df


def render_section(title: str, df: pd.DataFrame, expanded: bool = False) -> None:
    """Reusable dataframe section for future extension blocks."""
    with st.expander(title, expanded=expanded):
        if df.empty:
            st.info("Нет данных для отображения.")
        else:
            st.dataframe(df, use_container_width=True)


try:
    with get_connection() as conn:
        st.success(f"Подключено: {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['dbname']}")

        relations = list_public_relations(conn)
        required = {"moex_bonds_securities", "bond_emitters", "bond_emitters_market_view"}
        missing = sorted(required.difference(relations))
        if missing:
            st.warning("Отсутствуют обязательные объекты: " + ", ".join(missing))

        # Controls: filter/sort/limit + explicit load button (lazy loading).
        control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
        with control_col1:
            emitter_filter = st.text_input("Фильтр по названию эмитента", value="")
        with control_col2:
            sort_option = st.selectbox("Сортировка", list(SORT_OPTIONS.keys()), index=0)
        with control_col3:
            row_limit = st.number_input("LIMIT", min_value=10, max_value=50000, value=DEFAULT_LIMIT, step=10)

        load_clicked = st.button("Загрузить данные")

        # Lazy loading behavior:
        # - if a filter is entered -> auto-load filtered data;
        # - otherwise load only after explicit button click.
        should_load = bool(emitter_filter.strip()) or load_clicked

        if not should_load:
            st.info("Для быстрого старта: введите фильтр или нажмите 'Загрузить данные'.")
        else:
            all_emitters_df = fetch_emitters(
                conn=conn,
                emitter_filter=emitter_filter,
                sort_option=sort_option,
                limit=int(row_limit),
            )
            render_section("Все эмитенты", all_emitters_df, expanded=True)

        top10_count_df = fetch_top10(conn, "bonds_count")
        render_section("Топ-10 по количеству облигаций", top10_count_df)

        top10_maturity_df = fetch_top10(conn, "last_maturity_date")
        render_section("Топ-10 по дате погашения", top10_maturity_df)

except Exception as exc:
    st.error(f"Ошибка подключения/запроса к PostgreSQL: {exc}")
    st.info("Проверьте Docker PostgreSQL на 127.0.0.1:5433 и наличие нужных таблиц/представления.")
