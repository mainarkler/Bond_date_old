"""MOEX bonds loader -> PostgreSQL.

Performance notes:
- Uses batched UPSERT into `moex_bonds_securities`.
- Rebuilds `bond_emitters` once per loader run (not in Streamlit).
- Creates supporting indexes for fast aggregation and dashboard queries.
"""

import os
from typing import Dict, Iterable, List

import psycopg2
import requests
from psycopg2.extras import execute_batch

MOEX_SECURITIES_URL = "https://iss.moex.com/iss/securities.json"
BATCH_SIZE = 1000
PAGE_SIZE = 100


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "postgres"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "6f_@%DB&hA2+$f_"),
    )


def ensure_schema(conn) -> None:
    """Create required tables/view/indexes if they do not exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS moex_bonds_securities (
                secid TEXT PRIMARY KEY,
                shortname TEXT,
                name TEXT,
                isin TEXT,
                emitent_id TEXT,
                emitent_title TEXT,
                emitent_inn TEXT,
                issue_date DATE,
                maturity_date DATE,
                coupon_percent NUMERIC,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_moex_bonds_emitent_title ON moex_bonds_securities (emitent_title)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_moex_bonds_emitent_inn ON moex_bonds_securities (emitent_inn)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_moex_bonds_issue_date ON moex_bonds_securities (issue_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_moex_bonds_maturity_date ON moex_bonds_securities (maturity_date)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bond_emitters (
                emitter_key TEXT PRIMARY KEY,
                emitent_title TEXT,
                emitent_inn TEXT,
                bonds_count INTEGER NOT NULL,
                first_issue_date DATE,
                last_maturity_date DATE,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bond_emitters_title ON bond_emitters (emitent_title)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bond_emitters_bonds_count ON bond_emitters (bonds_count DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bond_emitters_last_maturity ON bond_emitters (last_maturity_date DESC)")

        cur.execute(
            """
            CREATE OR REPLACE VIEW bond_emitters_market_view AS
            SELECT
                emitent_title,
                emitent_inn,
                bonds_count,
                first_issue_date,
                last_maturity_date,
                updated_at
            FROM bond_emitters
            """
        )
    conn.commit()


def _normalize_row(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "secid": str(row.get("secid") or "").strip(),
        "shortname": row.get("shortname"),
        "name": row.get("name") or row.get("secname"),
        "isin": row.get("isin"),
        "emitent_id": str(row.get("emitent_id") or row.get("emitentid") or "").strip() or None,
        "emitent_title": row.get("emitent_title"),
        "emitent_inn": row.get("emitent_inn"),
        "issue_date": row.get("issuedate"),
        "maturity_date": row.get("matdate") or row.get("maturitydate"),
        "coupon_percent": row.get("couponpercent") or row.get("coupon_percent"),
    }


def fetch_bonds_from_moex() -> Iterable[Dict[str, object]]:
    start = 0
    while True:
        params = {
            "start": start,
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": (
                "secid,shortname,name,secname,isin,emitent_id,emitentid,emitent_title,"
                "emitent_inn,issuedate,matdate,maturitydate,couponpercent,group"
            ),
        }
        response = requests.get(MOEX_SECURITIES_URL, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json().get("securities", {})
        columns: List[str] = payload.get("columns", [])
        data_rows = payload.get("data", [])

        if not data_rows:
            break

        for values in data_rows:
            raw = dict(zip(columns, values))
            if "bond" not in str(raw.get("group") or "").lower():
                continue
            normalized = _normalize_row(raw)
            if normalized["secid"]:
                yield normalized

        start += PAGE_SIZE


def upsert_moex_bonds(conn, rows: Iterable[Dict[str, object]]) -> int:
    """Batched UPSERT into moex_bonds_securities."""
    sql_query = """
        INSERT INTO moex_bonds_securities (
            secid, shortname, name, isin, emitent_id, emitent_title,
            emitent_inn, issue_date, maturity_date, coupon_percent, updated_at
        ) VALUES (
            %(secid)s, %(shortname)s, %(name)s, %(isin)s, %(emitent_id)s, %(emitent_title)s,
            %(emitent_inn)s, %(issue_date)s, %(maturity_date)s, %(coupon_percent)s, NOW()
        )
        ON CONFLICT (secid) DO UPDATE SET
            shortname = EXCLUDED.shortname,
            name = EXCLUDED.name,
            isin = EXCLUDED.isin,
            emitent_id = EXCLUDED.emitent_id,
            emitent_title = EXCLUDED.emitent_title,
            emitent_inn = EXCLUDED.emitent_inn,
            issue_date = EXCLUDED.issue_date,
            maturity_date = EXCLUDED.maturity_date,
            coupon_percent = EXCLUDED.coupon_percent,
            updated_at = NOW();
    """

    total = 0
    batch: List[Dict[str, object]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                execute_batch(cur, sql_query, batch, page_size=BATCH_SIZE)
                total += len(batch)
                batch.clear()

        if batch:
            execute_batch(cur, sql_query, batch, page_size=BATCH_SIZE)
            total += len(batch)

    conn.commit()
    return total


def refresh_bond_emitters(conn) -> None:
    """Aggregate emitters once per loader run (not in Streamlit requests)."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE bond_emitters")
        cur.execute(
            """
            INSERT INTO bond_emitters (
                emitter_key,
                emitent_title,
                emitent_inn,
                bonds_count,
                first_issue_date,
                last_maturity_date,
                updated_at
            )
            SELECT
                COALESCE(NULLIF(emitent_inn, ''), NULLIF(emitent_id, ''), NULLIF(emitent_title, ''), 'UNKNOWN') AS emitter_key,
                COALESCE(NULLIF(emitent_title, ''), 'UNKNOWN') AS emitent_title,
                NULLIF(emitent_inn, '') AS emitent_inn,
                COUNT(*)::int AS bonds_count,
                MIN(issue_date) AS first_issue_date,
                MAX(maturity_date) AS last_maturity_date,
                NOW() AS updated_at
            FROM moex_bonds_securities
            GROUP BY 1, 2, 3
            """
        )
    conn.commit()


def main() -> None:
    conn = get_db_connection()
    try:
        ensure_schema(conn)
        loaded = upsert_moex_bonds(conn, fetch_bonds_from_moex())
        refresh_bond_emitters(conn)
        print(f"Loaded/updated bonds: {loaded}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
