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


def get_postgres_conn():
    """Create PostgreSQL connection from Streamlit secrets."""
    cfg = st.secrets["postgres"]
    return psycopg2.connect(
        host=cfg["host"],
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
        port=cfg["port"],
    )


def fetch_all_moex_bonds() -> pd.DataFrame:
    """Full MOEX bonds traversal (existing logic kept: load all bonds from ISS)."""
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "group_by": "group",
        "group_by_filter": "stock_bonds",
        "securities.columns": "secid,isin,shortname,emitent_title,primary_boardid",
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
                    "isin": row.get("isin"),
                    "shortname": row.get("shortname"),
                    "emitent_title": row.get("emitent_title"),
                    "primary_boardid": row.get("primary_boardid"),
                }
            )

        start += len(data)

    return pd.DataFrame(rows)


def upsert_moex_bonds_securities(conn, bonds_df: pd.DataFrame) -> None:
    if bonds_df.empty:
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BONDS_TABLE} (
                secid TEXT PRIMARY KEY,
                isin TEXT,
                shortname TEXT,
                emitent_title TEXT,
                primary_boardid TEXT,
                updated_at DATE NOT NULL DEFAULT CURRENT_DATE
            )
            """
        )

        values = [
            (
                str(row["secid"]) if pd.notna(row["secid"]) else None,
                str(row["isin"]) if pd.notna(row["isin"]) else None,
                str(row["shortname"]) if pd.notna(row["shortname"]) else None,
                str(row["emitent_title"]) if pd.notna(row["emitent_title"]) else None,
                str(row["primary_boardid"]) if pd.notna(row["primary_boardid"]) else None,
            )
            for _, row in bonds_df.iterrows()
            if pd.notna(row["secid"])
        ]

        execute_values(
            cur,
            f"""
            INSERT INTO {BONDS_TABLE} (secid, isin, shortname, emitent_title, primary_boardid)
            VALUES %s
            ON CONFLICT (secid) DO UPDATE SET
                isin = EXCLUDED.isin,
                shortname = EXCLUDED.shortname,
                emitent_title = EXCLUDED.emitent_title,
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
            sql.SQL("SELECT 1 FROM {} WHERE date = %s LIMIT 1").format(sql.Identifier(STAT_TABLE)),
            (today,),
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
            sql.SQL("SELECT 1 FROM {} WHERE date = %s LIMIT 1").format(sql.Identifier(STAT_TABLE)),
            (yesterday,),
        )
        yesterday_exists = cur.fetchone() is not None

        if not yesterday_exists:
            insert_query = sql.SQL(
                """
                INSERT INTO {} (date, emitent_title, bonds_count)
                VALUES %s
                ON CONFLICT (date, emitent_title) DO UPDATE
                SET bonds_count = EXCLUDED.bonds_count
                """
            ).format(sql.Identifier(STAT_TABLE))

            values = [(yesterday, emitter, count) for emitter, count in emitter_counts]
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
            INSERT INTO {} (date, emitent_title, bonds_count)
            VALUES %s
            ON CONFLICT (date, emitent_title) DO UPDATE
            SET bonds_count = EXCLUDED.bonds_count
            """
        ).format(sql.Identifier(STAT_TABLE))

        values = [(today, emitter, count) for emitter, count in emitter_counts]
        execute_values(cur, query.as_string(cur), values)

    conn.commit()


def run_daily_sync() -> None:
    conn = get_postgres_conn()
    try:
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
