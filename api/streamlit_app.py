"""Streamlit app: bond emitters dashboard backed by PostgreSQL.

The app is intentionally structured with small reusable functions so it can be
extended later with additional bond-market tables and analytics blocks.
"""

import os
from typing import List

import pandas as pd
import psycopg2
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Эмитенты облигаций", page_icon="📈", layout="wide")
st.title("📈 Эмитенты облигаций")
st.caption("Данные читаются из PostgreSQL (Docker) с последующей обработкой в Pandas.")


# -----------------------------------------------------------------------------
# DB connection config
# Priority: env vars first, then requested defaults
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "6f_@%DB&hA2+$f_"),
}

REQUIRED_RELATIONS = [
    "moex_bonds_securities",
    "bond_emitters",
    "bond_emitters_market_view",
]

SORT_OPTIONS = {
    "bonds_count (убыв.)": ("bonds_count", False),
    "bonds_count (возр.)": ("bonds_count", True),
    "last_maturity_date (убыв.)": ("last_maturity_date", False),
    "last_maturity_date (возр.)": ("last_maturity_date", True),
}


def get_connection() -> psycopg2.extensions.connection:
    """Create PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def list_available_relations(conn: psycopg2.extensions.connection) -> List[str]:
    """List tables and views from public schema."""
    sql = """
        SELECT table_name AS relation_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        UNION
        SELECT table_name AS relation_name
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY relation_name;
    """
    rel_df = pd.read_sql_query(sql, conn)
    return rel_df["relation_name"].tolist()


def load_emitters_market_data(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Load normalized emitters market data from public.bond_emitters_market_view."""
    sql = """
        SELECT
            emitent_title,
            emitent_inn,
            bonds_count,
            first_issue_date,
            last_maturity_date
        FROM public.bond_emitters_market_view
    """
    df = pd.read_sql_query(sql, conn)

    # Normalize expected types for stable filtering/sorting in UI.
    if "bonds_count" in df.columns:
        df["bonds_count"] = pd.to_numeric(df["bonds_count"], errors="coerce").fillna(0).astype(int)
    if "first_issue_date" in df.columns:
        df["first_issue_date"] = pd.to_datetime(df["first_issue_date"], errors="coerce").dt.date
    if "last_maturity_date" in df.columns:
        df["last_maturity_date"] = pd.to_datetime(df["last_maturity_date"], errors="coerce").dt.date

    return df


def apply_filters_and_sort(df: pd.DataFrame, emitter_filter: str, sort_choice: str) -> pd.DataFrame:
    """Apply emitter name filter and selected sorting option."""
    filtered = df.copy()

    if emitter_filter.strip():
        filtered = filtered[
            filtered["emitent_title"].fillna("").str.contains(emitter_filter.strip(), case=False, na=False)
        ]

    sort_column, ascending = SORT_OPTIONS[sort_choice]
    filtered = filtered.sort_values(by=sort_column, ascending=ascending, na_position="last")
    return filtered


def render_dataframe_section(title: str, df: pd.DataFrame, expanded: bool = False) -> None:
    """Render one expander section with dataframe."""
    with st.expander(title, expanded=expanded):
        if df.empty:
            st.info("Нет данных для отображения.")
        else:
            st.dataframe(df, use_container_width=True)


try:
    with get_connection() as conn:
        st.success(f"Подключено к PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['dbname']}")

        available_relations = list_available_relations(conn)
        st.write("Найденные таблицы/представления в schema public:")
        st.code("\n".join(available_relations) if available_relations else "(пусто)")

        missing = [name for name in REQUIRED_RELATIONS if name not in available_relations]
        if missing:
            st.warning(
                "Отсутствуют требуемые объекты: " + ", ".join(missing) + ". "
                "Проверьте загрузчик/миграции БД."
            )

        market_df = load_emitters_market_data(conn)

        # Filters and sorting controls.
        controls_col1, controls_col2 = st.columns(2)
        with controls_col1:
            emitter_filter = st.text_input("Фильтр по названию эмитента", value="")
        with controls_col2:
            sort_choice = st.selectbox("Сортировка", list(SORT_OPTIONS.keys()), index=0)

        result_df = apply_filters_and_sort(market_df, emitter_filter, sort_choice)

        # Section 1: all emitters (with filter/sort applied)
        render_dataframe_section("Все эмитенты (с фильтром и сортировкой)", result_df, expanded=True)

        # Section 2: top 10 by bond count
        top10_df = (
            market_df.sort_values(by="bonds_count", ascending=False, na_position="last")
            .head(10)
            .reset_index(drop=True)
        )
        render_dataframe_section("Топ-10 эмитентов по количеству облигаций", top10_df)

        # Section 3: rows with the latest maturity dates
        last_maturity_df = (
            market_df.sort_values(by="last_maturity_date", ascending=False, na_position="last")
            .head(10)
            .reset_index(drop=True)
        )
        render_dataframe_section("Топ-10 по последней дате погашения", last_maturity_df)

except Exception as exc:
    st.error(f"Ошибка подключения/чтения PostgreSQL: {exc}")
    st.info("Проверьте, что Docker PostgreSQL доступен на 127.0.0.1:5433 и содержит нужные таблицы.")
