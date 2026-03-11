import os
from typing import Dict, Iterable, List

import psycopg2
import requests
from psycopg2.extras import execute_batch

MOEX_SECURITIES_URL = "https://iss.moex.com/iss/securities.json"
BATCH_SIZE = 100
PAGE_SIZE = 100


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "marketdata"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
    )


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bonds (
                secid TEXT PRIMARY KEY,
                shortname TEXT,
                name TEXT,
                isin TEXT,
                emitent_id TEXT,
                emitent_title TEXT,
                emitent_inn TEXT,
                coupon_percent NUMERIC,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
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
        "coupon_percent": row.get("couponpercent") or row.get("coupon_percent"),
    }


def fetch_bonds_from_moex() -> Iterable[Dict[str, object]]:
    start = 0

    while True:
        params = {
            "start": start,
            "iss.meta": "off",
            "iss.only": "securities",
            "securities.columns": "secid,shortname,name,secname,isin,emitent_id,emitentid,emitent_title,emitent_inn,couponpercent,group",
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
            group_value = str(raw.get("group") or "").lower()
            if "bond" not in group_value:
                continue

            normalized = _normalize_row(raw)
            if normalized["secid"]:
                yield normalized

        start += PAGE_SIZE


def upsert_bonds(conn, rows: Iterable[Dict[str, object]]) -> int:
    batch = []
    total = 0

    sql = """
        INSERT INTO bonds (
            secid,
            shortname,
            name,
            isin,
            emitent_id,
            emitent_title,
            emitent_inn,
            coupon_percent,
            updated_at
        )
        VALUES (
            %(secid)s,
            %(shortname)s,
            %(name)s,
            %(isin)s,
            %(emitent_id)s,
            %(emitent_title)s,
            %(emitent_inn)s,
            %(coupon_percent)s,
            NOW()
        )
        ON CONFLICT (secid) DO UPDATE SET
            shortname = EXCLUDED.shortname,
            name = EXCLUDED.name,
            isin = EXCLUDED.isin,
            emitent_id = EXCLUDED.emitent_id,
            emitent_title = EXCLUDED.emitent_title,
            emitent_inn = EXCLUDED.emitent_inn,
            coupon_percent = EXCLUDED.coupon_percent,
            updated_at = NOW();
    """

    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                execute_batch(cur, sql, batch)
                total += len(batch)
                batch.clear()

        if batch:
            execute_batch(cur, sql, batch)
            total += len(batch)

    conn.commit()
    return total


def main() -> None:
    conn = get_db_connection()
    try:
        ensure_schema(conn)
        loaded = upsert_bonds(conn, fetch_bonds_from_moex())
        print(f"Loaded/updated bonds: {loaded}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
