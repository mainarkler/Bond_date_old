from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import psycopg2
import requests
import streamlit as st
from psycopg2 import sql
from psycopg2.extras import execute_values

MOEX_BONDS_URL = "https://iss.moex.com/iss/securities.json"
STAT_TABLE = "Статистика рынка"
BONDS_TABLE = "moex_bonds_securities"
MARKET_TYPE = "bonds"


def get_postgres_conn():
    """Create PostgreSQL connection from Streamlit secrets."""
    raw_pg = st.secrets.get("postgres")
    flat = st.secrets

    db_url = ""
    if isinstance(raw_pg, str):
        db_url = raw_pg.strip()
    elif raw_pg is not None:
        db_url = str(raw_pg.get("url", "")).strip()

    if not db_url:
        db_url = str(flat.get("DB_URL", flat.get("DATABASE_URL", flat.get("POSTGRES_URL", "")))).strip()

    if db_url:
        return psycopg2.connect(db_url)

    raise RuntimeError(
        "Не найден DB_URL для PostgreSQL. Укажите st.secrets['postgres']['url'] или DB_URL в secrets.toml"
    )


def fetch_all_moex_bonds() -> pd.DataFrame:
    """Full MOEX bonds traversal (existing logic kept: load all bonds from ISS)."""
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "group_by": "group",
        "group_by_filter": "stock_bonds",
        "securities.columns": "secid,shortname,name,isin,emitent_id,emitent_title,emitent_inn,primary_boardid",
        "start": 0,
        "limit": 100,
    }

    rows: list[dict] = []
    start = 0

    while True:
        params["start"] = start
        response = requests.get(MOEX_BONDS_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json().get("securities", {})
        columns = payload.get("columns", [])
        data = payload.get("data", [])

        if not data:
            break

        for item in data:
            row = dict(zip(columns, item))
            rows.append(
                {
                    "secid": row.get("secid"),
                    "shortname": row.get("shortname"),
                    "name": row.get("name"),
                    "isin": row.get("isin"),
                    "emitent_id": row.get("emitent_id"),
                    "emitent_title": row.get("emitent_title"),
                    "emitent_inn": row.get("emitent_inn"),
                    "primary_boardid": row.get("primary_boardid"),
                }
            )

        start += len(data)

    return pd.DataFrame(rows)


def ensure_statistics_table_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    date DATE NOT NULL,
                    emitent_title TEXT NOT NULL,
                    bonds_count INTEGER NOT NULL DEFAULT 0,
                    market_type TEXT NOT NULL DEFAULT 'bonds'
                )
                """
            ).format(sql.Identifier(STAT_TABLE))
        )
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS market_type TEXT NOT NULL DEFAULT 'bonds'").format(
                sql.Identifier(STAT_TABLE)
            )
        )
        cur.execute(
            sql.SQL(
                "CREATE UNIQUE INDEX IF NOT EXISTS market_statistics_uq ON {} (date, emitent_title, market_type)"
            ).format(sql.Identifier(STAT_TABLE))
        )
    conn.commit()


def upsert_moex_bonds_securities(conn, bonds_df: pd.DataFrame) -> None:
    if bonds_df.empty:
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BONDS_TABLE} (
                secid TEXT PRIMARY KEY,
                shortname TEXT,
                name TEXT,
                isin TEXT,
                emitent_id TEXT,
                emitent_title TEXT,
                emitent_inn TEXT,
                primary_boardid TEXT,
                updated_at DATE NOT NULL DEFAULT CURRENT_DATE
            )
            """
        )
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS shortname TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS name TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS isin TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS emitent_id TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS emitent_title TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS emitent_inn TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS primary_boardid TEXT")
        cur.execute(f"ALTER TABLE {BONDS_TABLE} ADD COLUMN IF NOT EXISTS updated_at DATE NOT NULL DEFAULT CURRENT_DATE")

        values = [
            (
                str(row["secid"]) if pd.notna(row["secid"]) else None,
                str(row["shortname"]) if pd.notna(row["shortname"]) else None,
                str(row["name"]) if pd.notna(row["name"]) else None,
                str(row["isin"]) if pd.notna(row["isin"]) else None,
                str(row["emitent_id"]) if pd.notna(row["emitent_id"]) else None,
                str(row["emitent_title"]) if pd.notna(row["emitent_title"]) else None,
                str(row["emitent_inn"]) if pd.notna(row["emitent_inn"]) else None,
                str(row["primary_boardid"]) if pd.notna(row["primary_boardid"]) else None,
            )
            for _, row in bonds_df.iterrows()
            if pd.notna(row["secid"])
        ]

        execute_values(
            cur,
            f"""
            INSERT INTO {BONDS_TABLE} (
                secid, shortname, name, isin, emitent_id, emitent_title, emitent_inn, primary_boardid
            )
            VALUES %s
            ON CONFLICT (secid) DO UPDATE SET
                shortname = EXCLUDED.shortname,
                name = EXCLUDED.name,
                isin = EXCLUDED.isin,
                emitent_id = EXCLUDED.emitent_id,
                emitent_title = EXCLUDED.emitent_title,
                emitent_inn = EXCLUDED.emitent_inn,
                primary_boardid = EXCLUDED.primary_boardid,
                updated_at = CURRENT_DATE
            """,
            values,
        )

    conn.commit()


def is_first_run_today(conn) -> bool:
    today = date.today()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT 1 FROM {} WHERE date = %s AND market_type = %s LIMIT 1").format(sql.Identifier(STAT_TABLE)),
            (today, MARKET_TYPE),
        )
        exists = cur.fetchone() is not None
    return not exists


def load_bonds_count_by_emitter(conn) -> Iterable[tuple[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT emitent_title, COUNT(*)::INT AS bonds_count
            FROM {BONDS_TABLE}
            WHERE emitent_title IS NOT NULL AND TRIM(emitent_title) <> ''
            GROUP BY emitent_title
            """
        )
        rows = cur.fetchall()
    return rows


def ensure_yesterday_statistics(conn, emitter_counts: Iterable[tuple[str, int]]) -> None:
    yesterday = date.today() - timedelta(days=1)
    emitter_counts = list(emitter_counts)
    if not emitter_counts:
        return

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT 1 FROM {} WHERE date = %s AND market_type = %s LIMIT 1").format(sql.Identifier(STAT_TABLE)),
            (yesterday, MARKET_TYPE),
        )
        yesterday_exists = cur.fetchone() is not None

        if not yesterday_exists:
            insert_query = sql.SQL(
                """
                INSERT INTO {} (date, emitent_title, bonds_count, market_type)
                VALUES %s
                ON CONFLICT (date, emitent_title, market_type) DO UPDATE
                SET bonds_count = EXCLUDED.bonds_count
                """
            ).format(sql.Identifier(STAT_TABLE))

            values = [(yesterday, emitter, count, MARKET_TYPE) for emitter, count in emitter_counts]
            execute_values(cur, insert_query.as_string(cur), values)

    conn.commit()


def update_today_bonds_count(conn, emitter_counts: Iterable[tuple[str, int]]) -> None:
    today = date.today()
    emitter_counts = list(emitter_counts)
    if not emitter_counts:
        return

    with conn.cursor() as cur:
        query = sql.SQL(
            """
            INSERT INTO {} (date, emitent_title, bonds_count, market_type)
            VALUES %s
            ON CONFLICT (date, emitent_title, market_type) DO UPDATE
            SET bonds_count = EXCLUDED.bonds_count
            """
        ).format(sql.Identifier(STAT_TABLE))

        values = [(today, emitter, count, MARKET_TYPE) for emitter, count in emitter_counts]
        execute_values(cur, query.as_string(cur), values)

    conn.commit()


def run_daily_sync() -> None:
    conn = get_postgres_conn()
    try:
        ensure_statistics_table_schema(conn)
        bonds_df = fetch_all_moex_bonds()
        upsert_moex_bonds_securities(conn, bonds_df)

        if is_first_run_today(conn):
            emitter_counts = load_bonds_count_by_emitter(conn)
            ensure_yesterday_statistics(conn, emitter_counts)
            update_today_bonds_count(conn, emitter_counts)
    finally:
        conn.close()


if __name__ == "__main__":
    run_daily_sync()
