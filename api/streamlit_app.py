import os

import pandas as pd
import psycopg2
import streamlit as st


st.set_page_config(page_title="Market Statistics", page_icon="📈", layout="wide")
st.title("📈 Market Statistics")
st.caption("Данные читаются только из PostgreSQL, наполняемого loader/moex_loader.py")


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "marketdata"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
    )


def query_df(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


try:
    with get_connection() as conn:
        total_bonds = query_df(conn, "SELECT COUNT(*) AS total_bonds FROM bonds")
        unique_issuers = query_df(
            conn,
            """
            SELECT COUNT(DISTINCT COALESCE(NULLIF(emitent_title, ''), emitent_id)) AS unique_issuers
            FROM bonds
            """,
        )
        avg_coupon = query_df(
            conn,
            "SELECT ROUND(AVG(coupon_percent)::numeric, 4) AS avg_coupon FROM bonds WHERE coupon_percent IS NOT NULL",
        )
        distribution = query_df(
            conn,
            """
            SELECT
                COALESCE(NULLIF(emitent_title, ''), emitent_id, 'UNKNOWN') AS issuer,
                COUNT(*) AS bonds_count
            FROM bonds
            GROUP BY 1
            ORDER BY bonds_count DESC, issuer
            """,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total number of bonds", int(total_bonds.loc[0, "total_bonds"]))
        col2.metric("Number of unique issuers", int(unique_issuers.loc[0, "unique_issuers"]))

        avg_coupon_value = avg_coupon.loc[0, "avg_coupon"]
        col3.metric("Average coupon", f"{avg_coupon_value}%" if pd.notna(avg_coupon_value) else "N/A")

        st.subheader("Distribution by issuer")
        if distribution.empty:
            st.info("No rows in bonds table. Run: python loader/moex_loader.py")
        else:
            st.bar_chart(distribution.set_index("issuer")["bonds_count"], use_container_width=True)
            st.dataframe(distribution, use_container_width=True)

except Exception as exc:
    st.error(f"Database query failed: {exc}")
    st.info("Проверьте DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD и запустите loader.")
