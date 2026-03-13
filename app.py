import csv
import os
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import requests
import sell_stress as ss
import streamlit as st
import index_analytics as ia
from email_compose import render_email_compose_section
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from services.moex_turnover import MoexTurnoverClient

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="РЕПО претрейд", page_icon="📈", layout="wide")
st.title("Stat bord")

# ---------------------------
# Session state defaults
# ---------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None
if "file_loaded" not in st.session_state:
    st.session_state["file_loaded"] = False
if "last_file_name" not in st.session_state:
    st.session_state["last_file_name"] = None
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "home"
if "vm_last_report" not in st.session_state:
    st.session_state["vm_last_report"] = None
if "calendar_last_report" not in st.session_state:
    st.session_state["calendar_last_report"] = None

FORCED_ACTIVE_VIEW = os.getenv("FORCE_ACTIVE_VIEW", "").strip().lower()
if FORCED_ACTIVE_VIEW in {
    "repo",
    "calendar",
    "vm",
    "sell_stres",
    "index_analytics",
    "moex_turnover",
    "market_statistics",
    "turnover_export",
}:
    st.session_state["active_view"] = FORCED_ACTIVE_VIEW

# ---------------------------
# Main navigation
# ---------------------------
def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def init_sell_stres_state():
    defaults = {
        "sell_stres_share_downloads": None,
        "sell_stres_share_show_tables": False,
        "sell_stres_share_table_results": {},
        "sell_stres_share_meta_table": None,
        "sell_stres_bond_downloads": None,
        "sell_stres_bond_show_tables": False,
        "sell_stres_bond_table_results": {},
        "sell_stres_bond_meta_table": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


if st.session_state["active_view"] != "home" and not FORCED_ACTIVE_VIEW:
    if st.button("⬅️ На главную"):
        st.session_state["active_view"] = "home"
        trigger_rerun()

if st.session_state["active_view"] == "home":
    st.subheader("")
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("### Претрейд РЕПО")
        st.caption("Анализ ISIN и ключевых дат бумаг для сделок РЕПО.")
        if st.button("Открыть", key="open_repo", use_container_width=True):
            st.session_state["active_view"] = "repo"
            trigger_rerun()
    with top_right:
        st.markdown("### Календарь выплат")
        st.caption("Загрузка портфеля и построение календаря купонов и погашений.")
        if st.button("Открыть", key="open_calendar", use_container_width=True):
            st.session_state["active_view"] = "calendar"
            trigger_rerun()
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("### Расчет VM")
        st.caption("Расчет вариационной маржи по фьючерсам FORTS.")
        if st.button("Открыть", key="open_vm", use_container_width=True):
            st.session_state["active_view"] = "vm"
            trigger_rerun()

        st.markdown("### Состав индекса")
        st.caption("Загрузка состава индекса по датам и построение матрицы весов.")
        if st.button("Открыть", key="open_index_analytics", use_container_width=True):
            st.session_state["active_view"] = "index_analytics"
            trigger_rerun()

        st.markdown("### Выгрузка оборотов")
        st.caption("Обороты акций/облигаций за период с опцией NDM и Excel-отчётом.")
        if st.button("Открыть", key="open_turnover_export_home", use_container_width=True):
            st.session_state["active_view"] = "turnover_export"
            trigger_rerun()
    with bottom_right:
        st.markdown("### Sell_stress")
        st.caption("Оценка рыночного давления для акций и облигаций.")
        if st.button("Открыть", key="open_sell_stres", use_container_width=True):
            st.session_state["active_view"] = "sell_stres"
            trigger_rerun()

        st.markdown("### MOEX turnover")
        st.caption("Расчет оборота по сделкам MOEX ISS trades: regular / SPEQ / NDM.")
        if st.button("Открыть", key="open_moex_turnover", use_container_width=True):
            st.session_state["active_view"] = "moex_turnover"
            trigger_rerun()

        st.markdown("### Статистика рынка")
        st.caption("История объема торгов по акциям/облигациям с графиками и выгрузкой.")
        if st.button("Открыть", key="open_market_statistics", use_container_width=True):
            st.session_state["active_view"] = "market_statistics"
            trigger_rerun()
    st.stop()

# ---------------------------
# HTTP session with retries
# ---------------------------
def build_http_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "python-requests/iss-moex-script"})
    return session


HTTP_SESSION = build_http_session()


def request_get(url: str, timeout: int = 15, params=None):
    response = HTTP_SESSION.get(url, timeout=timeout, params=params)
    response.raise_for_status()
    return response


def request_json(url: str, timeout: int = 15, params=None, attempts: int = 3) -> dict:
    last_exc = None
    for _ in range(max(1, attempts)):
        response = request_get(url, timeout=timeout, params=params)
        try:
            return response.json()
        except ValueError as exc:
            last_exc = exc
            body_preview = (response.text or "")[:200].replace("\n", " ")
            if body_preview:
                continue
    raise RuntimeError(
        f"MOEX ISS вернул не-JSON для {url}. Последняя ошибка: {last_exc}"
    ) from last_exc


def open_index_analytics_sheet():
    ia.render_index_analytics_view(
        request_get=request_get,
        dataframe_to_excel_bytes=ss.dataframe_to_excel_bytes,
    )


def resolve_market_security_profile(identifier: str, market_kind: str) -> dict:
    normalized = identifier.strip().upper()
    if not normalized:
        raise ValueError("Пустой идентификатор")

    response = request_get(
        "https://iss.moex.com/iss/securities.json",
        params={"q": normalized, "iss.meta": "off"},
        timeout=200,
    )
    js = response.json()
    df = pd.DataFrame(js["securities"]["data"], columns=js["securities"]["columns"])
    if df.empty:
        raise ValueError(f"Инструмент '{identifier}' не найден на MOEX")

    if "group" in df.columns:
        if market_kind == "shares":
            df_filtered = df[df["group"].astype(str).str.contains("share", case=False, na=False)]
        else:
            df_filtered = df[df["group"].astype(str).str.contains("bond", case=False, na=False)]
        if not df_filtered.empty:
            df = df_filtered

    if "secid" in df.columns:
        exact_secid = df[df["secid"].astype(str).str.upper() == normalized]
        if not exact_secid.empty:
            row = exact_secid.iloc[0]
        elif "isin" in df.columns:
            exact_isin = df[df["isin"].astype(str).str.upper() == normalized]
            row = exact_isin.iloc[0] if not exact_isin.empty else df.iloc[0]
        else:
            row = df.iloc[0]
    else:
        row = df.iloc[0]

    secid = str(row.get("secid", "")).strip().upper()
    if not secid:
        raise ValueError(f"Не удалось определить SECID для '{identifier}'")

    return {
        "input": identifier,
        "secid": secid,
        "isin": str(row.get("isin", "")).strip().upper(),
        "shortname": str(row.get("shortname", "")).strip(),
        "emitent_title": str(row.get("emitent_title", "")).strip(),
    }


def load_market_history_values(secid: str, market_kind: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = 0
    all_rows = []
    columns = []
    while True:
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/{market_kind}/securities/{secid}.json"
        response = request_get(
            url,
            params={
                "from": start_date,
                "till": end_date,
                "start": start,
                "iss.only": "history",
                "iss.meta": "off",
                "history.columns": "TRADEDATE,VALUE,NUMTRADES,VOLUME,BOARDID,SHORTNAME,SECID",
            },
            timeout=200,
        )
        payload = response.json().get("history", {})
        rows = payload.get("data", [])
        columns = payload.get("columns", columns)
        if not rows:
            break
        all_rows.extend(rows)
        start += len(rows)

    if not all_rows:
        return pd.DataFrame(columns=["TRADEDATE", "VALUE", "NUMTRADES", "VOLUME", "SECID", "SHORTNAME"])

    df = pd.DataFrame(all_rows, columns=columns)
    for col in ["VALUE", "NUMTRADES", "VOLUME"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"])

    agg = (
        df.groupby(["TRADEDATE", "SECID", "SHORTNAME"], as_index=False)
        .agg(VALUE=("VALUE", "sum"), NUMTRADES=("NUMTRADES", "sum"), VOLUME=("VOLUME", "sum"))
        .sort_values(["TRADEDATE", "SECID"])
    )
    return agg


def load_market_wide_history_values(market_kind: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = 0
    all_rows = []
    columns = []
    while True:
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/{market_kind}/securities.json"
        response = request_get(
            url,
            params={
                "from": start_date,
                "till": end_date,
                "start": start,
                "iss.only": "history",
                "iss.meta": "off",
                "history.columns": "TRADEDATE,VALUE,NUMTRADES,VOLUME,SHORTNAME,SECID",
            },
            timeout=200,
        )
        payload = response.json().get("history", {})
        rows = payload.get("data", [])
        columns = payload.get("columns", columns)
        if not rows:
            break
        all_rows.extend(rows)
        start += len(rows)

    if not all_rows:
        return pd.DataFrame(columns=["TRADEDATE", "VALUE", "NUMTRADES", "VOLUME", "SECID", "SHORTNAME"])

    df = pd.DataFrame(all_rows, columns=columns)
    for col in ["VALUE", "NUMTRADES", "VOLUME"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"])

    return (
        df.groupby(["TRADEDATE", "SECID", "SHORTNAME"], as_index=False)
        .agg(VALUE=("VALUE", "sum"), NUMTRADES=("NUMTRADES", "sum"), VOLUME=("VOLUME", "sum"))
        .sort_values(["TRADEDATE", "SECID"])
    )


def load_security_emitents_map(market_kind: str) -> pd.DataFrame:
    start = 0
    rows_all = []
    columns = []

    while True:
        response = request_get(
            "https://iss.moex.com/iss/securities.json",
            params={
                "start": start,
                "iss.meta": "off",
                "iss.only": "securities",
                "securities.columns": "secid,group,emitent_title",
            },
            timeout=200,
        )
        payload = response.json().get("securities", {})
        rows = payload.get("data", [])
        columns = payload.get("columns", columns)
        if not rows:
            break
        rows_all.extend(rows)
        start += len(rows)

    if not rows_all:
        return pd.DataFrame(columns=["SECID", "EMITENT_TITLE"])

    df = pd.DataFrame(rows_all, columns=columns)
    if "group" in df.columns:
        if market_kind == "shares":
            df = df[df["group"].astype(str).str.contains("share", case=False, na=False)]
        else:
            df = df[df["group"].astype(str).str.contains("bond", case=False, na=False)]
    if df.empty:
        return pd.DataFrame(columns=["SECID", "EMITENT_TITLE"])

    df = df.rename(columns={"secid": "SECID", "emitent_title": "EMITENT_TITLE"})
    df["SECID"] = df["SECID"].astype(str).str.upper()
    df["EMITENT_TITLE"] = df["EMITENT_TITLE"].astype(str).replace("nan", "", regex=False)
    return df[["SECID", "EMITENT_TITLE"]].drop_duplicates(subset=["SECID"])


def normalize_emitent_title(value: str) -> str:
    title = str(value or "").strip()
    return title if title else "Не указан"


def calculate_turnover_liquidity_stats(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["SECID", "ADTV", "MDTV", "SIGMA"])

    work_df = history_df.copy()
    work_df["VALUE"] = pd.to_numeric(work_df["VALUE"], errors="coerce").fillna(0.0)

    return (
        work_df.groupby("SECID", as_index=False)
        .agg(ADTV=("VALUE", "mean"), MDTV=("VALUE", "median"), SIGMA=("VALUE", "std"))
        .fillna(0.0)
    )


def load_turnover_components_via_iss(
    secid: str,
    market_kind: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    boards_js = request_json(
        f"https://iss.moex.com/iss/securities/{secid}/boards.json",
        params={"iss.meta": "off"},
        timeout=200,
    )
    boards_df = pd.DataFrame(boards_js["boards"]["data"], columns=boards_js["boards"]["columns"])
    if boards_df.empty:
        empty_daily = pd.DataFrame(columns=["SECID", "TRADEDATE", "REGULAR", "SPEQ", "NDM", "TOTAL_TRADES"])
        return empty_daily, {"TOTAL_regular": 0.0, "TOTAL_SPEQ": 0.0, "TOTAL_NDM": 0.0, "TOTAL_all": 0.0}

    boards_df = boards_df[boards_df["is_traded"] == 1].copy()
    regular_boards = boards_df[
        (boards_df["market"].astype(str) == market_kind)
        & (boards_df["boardid"].astype(str).str.upper() != "SPEQ")
    ][["boardid"]].drop_duplicates()
    speq_boards = boards_df[boards_df["boardid"].astype(str).str.upper() == "SPEQ"][["boardid"]].drop_duplicates()
    ndm_boards = boards_df[
        boards_df["market"].astype(str).str.contains("ndm", case=False, na=False)
    ][["boardid"]].drop_duplicates()

    def load_category(board_pairs: pd.DataFrame, category_name: str) -> pd.DataFrame:
        cat_rows = []
        for row in board_pairs.itertuples(index=False):
            start = 0
            while True:
                payload = request_json(
                    f"https://iss.moex.com/iss/history/engines/stock/markets/{market_kind}/securities/{secid}.json",
                    params={
                        "from": start_date,
                        "till": end_date,
                        "start": start,
                        "iss.only": "history",
                        "iss.meta": "off",
                        "history.columns": "TRADEDATE,BOARDID,VALUE",
                    },
                    timeout=200,
                ).get("history", {})
                data = payload.get("data", [])
                cols = payload.get("columns", [])
                if not data:
                    break

                df_part = pd.DataFrame(data, columns=cols)
                if "BOARDID" in df_part.columns:
                    df_part = df_part[df_part["BOARDID"].astype(str).str.upper() == str(row.boardid).upper()]
                if not df_part.empty and {"TRADEDATE", "VALUE"}.issubset(df_part.columns):
                    df_part = df_part[["TRADEDATE", "VALUE"]].copy()
                    df_part[category_name] = pd.to_numeric(df_part["VALUE"], errors="coerce").fillna(0.0)
                    cat_rows.append(df_part[["TRADEDATE", category_name]])
                start += len(data)

        if not cat_rows:
            return pd.DataFrame(columns=["TRADEDATE", category_name])
        cat_df = pd.concat(cat_rows, ignore_index=True)
        return cat_df.groupby("TRADEDATE", as_index=False)[category_name].sum()

    reg_df = load_category(regular_boards, "REGULAR")
    speq_df = load_category(speq_boards, "SPEQ")
    ndm_df = load_category(ndm_boards, "NDM")

    all_dates = pd.DataFrame(columns=["TRADEDATE"])
    for frame in [reg_df, speq_df, ndm_df]:
        if not frame.empty:
            all_dates = pd.concat([all_dates, frame[["TRADEDATE"]]], ignore_index=True)

    if all_dates.empty:
        daily_df = pd.DataFrame(columns=["SECID", "TRADEDATE", "REGULAR", "SPEQ", "NDM", "TOTAL_TRADES"])
    else:
        daily_df = all_dates.drop_duplicates().merge(reg_df, on="TRADEDATE", how="left")
        daily_df = daily_df.merge(speq_df, on="TRADEDATE", how="left")
        daily_df = daily_df.merge(ndm_df, on="TRADEDATE", how="left")
        daily_df[["REGULAR", "SPEQ", "NDM"]] = daily_df[["REGULAR", "SPEQ", "NDM"]].fillna(0.0)
        daily_df["TOTAL_TRADES"] = daily_df[["REGULAR", "SPEQ", "NDM"]].sum(axis=1)
        daily_df.insert(0, "SECID", secid)

    total_regular = float(pd.to_numeric(daily_df.get("REGULAR", 0.0), errors="coerce").sum()) if not daily_df.empty else 0.0
    total_speq = float(pd.to_numeric(daily_df.get("SPEQ", 0.0), errors="coerce").sum()) if not daily_df.empty else 0.0
    total_ndm = float(pd.to_numeric(daily_df.get("NDM", 0.0), errors="coerce").sum()) if not daily_df.empty else 0.0

    totals = {
        "TOTAL_regular": total_regular,
        "TOTAL_SPEQ": total_speq,
        "TOTAL_NDM": total_ndm,
        "TOTAL_all": total_regular + total_speq + total_ndm,
    }
    return daily_df, totals


# ---------------------------
# Sell_stres helpers (Share)
# ---------------------------
BASE_HISTORY_URL = (
    "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
)


def isin_to_secid(isin: str) -> str:
    params = {"q": isin, "iss.meta": "off"}
    response = request_get("https://iss.moex.com/iss/securities.json", params=params, timeout=200)
    js = response.json()
    df = pd.DataFrame(js["securities"]["data"], columns=js["securities"]["columns"])
    df = df[df["isin"] == isin]
    if df.empty:
        raise ValueError(f"ISIN {isin} не найден на MOEX")
    return df["secid"].iloc[0]


def resolve_share_identifier_to_isin(identifier: str) -> str | None:
    normalized = identifier.strip().upper()
    if not normalized:
        return None
    if isin_format_valid(normalized) and isin_checksum_valid(normalized):
        return normalized

    params = {"q": normalized, "iss.meta": "off"}
    response = request_get("https://iss.moex.com/iss/securities.json", params=params, timeout=200)
    js = response.json()
    df = pd.DataFrame(js["securities"]["data"], columns=js["securities"]["columns"])
    if df.empty or "isin" not in df.columns:
        return None

    if "secid" in df.columns:
        exact_secid = df[df["secid"].astype(str).str.upper() == normalized]
        exact_secid = exact_secid[exact_secid["isin"].notna() & (exact_secid["isin"].astype(str).str.strip() != "")]
        if not exact_secid.empty:
            return str(exact_secid.iloc[0]["isin"]).strip().upper()

    with_isin = df[df["isin"].notna() & (df["isin"].astype(str).str.strip() != "")]
    if with_isin.empty:
        return None
    return str(with_isin.iloc[0]["isin"]).strip().upper()


def resolve_identifier_to_secid(identifier: str) -> str | None:
    normalized = identifier.strip().upper()
    if not normalized:
        return None
    if isin_format_valid(normalized) and isin_checksum_valid(normalized):
        return isin_to_secid(normalized)

    params = {"q": normalized, "iss.meta": "off"}
    response = request_get("https://iss.moex.com/iss/securities.json", params=params, timeout=200)
    js = response.json()
    df = pd.DataFrame(js["securities"]["data"], columns=js["securities"]["columns"])
    if df.empty or "secid" not in df.columns:
        return None

    exact = df[df["secid"].astype(str).str.upper() == normalized]
    if not exact.empty:
        return str(exact.iloc[0]["secid"]).strip().upper()

    with_secid = df[df["secid"].notna() & (df["secid"].astype(str).str.strip() != "")]
    if with_secid.empty:
        return None
    return str(with_secid.iloc[0]["secid"]).strip().upper()


def load_moex_history(secid: str) -> pd.DataFrame:
    start = 0
    all_rows = []
    while True:
        url = f"{BASE_HISTORY_URL}/{secid}.json"
        response = request_get(url, params={"start": start}, timeout=200)
        js = response.json()
        rows = js["history"]["data"]
        cols = js["history"]["columns"]
        if not rows:
            break
        all_rows.extend(rows)
        start += len(rows)
    return pd.DataFrame(all_rows, columns=cols)


def build_q_vector(mode: str, q_max: int, q_step: int = 10_000, q_points: int = 200) -> np.ndarray:
    q_max = int(q_max)
    if q_max <= 0:
        raise ValueError("Q должен быть положительным")
    if mode == "linear":
        return np.arange(1, q_max + q_step, q_step, dtype=np.int64)
    if mode == "log":
        q = np.logspace(0, np.log10(q_max), q_points)
        return np.unique(np.round(q).astype(np.int64))
    raise ValueError("Режим Q должен быть 'linear' или 'log'")


def calculate_share_delta_p(
    isin: str,
    c_value: float,
    date_from: str,
    q_values: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    secid = isin_to_secid(isin)
    df = load_moex_history(secid)
    df = df[["TRADEDATE", "SECID", "HIGH", "LOW", "CLOSE", "VALUE"]].copy()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    num_cols = ["HIGH", "LOW", "CLOSE", "VALUE"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    date_from_dt = pd.to_datetime(date_from)
    date_to_dt = pd.to_datetime(datetime.now().date() - timedelta(days=1))
    df = df[(df["TRADEDATE"] >= date_from_dt) & (df["TRADEDATE"] <= date_to_dt)].dropna(
        subset=num_cols
    )
    if df.empty:
        raise ValueError("Нет данных после фильтрации по датам")

    df_day = (
        df.groupby(["TRADEDATE", "SECID"], as_index=False)
        .agg(
            HIGH=("HIGH", "mean"),
            LOW=("LOW", "mean"),
            CLOSE=("CLOSE", "mean"),
            VALUE=("VALUE", "sum"),
        )
        .sort_values("TRADEDATE")
    )
    t_len = len(df_day)
    if t_len == 0:
        raise ValueError("Пустой период наблюдений")

    sigma = np.sqrt(((df_day["HIGH"] - df_day["LOW"]) / df_day["CLOSE"]).sum() / t_len)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Некорректное σ = {sigma}")

    mdtv = np.median(df_day["VALUE"])
    if not np.isfinite(mdtv) or mdtv <= 0:
        raise ValueError(f"Некорректный MDTV = {mdtv}")

    delta_p = c_value * sigma * np.sqrt(q_values / mdtv)
    result = pd.DataFrame({"Q": q_values, "DeltaP": delta_p})
    meta = {
        "ISIN": isin,
        "T": t_len,
        "Sigma": float(sigma),
        "MDTV": float(mdtv),
    }
    return result, meta


def generate_q(mode: str, q_max: int, points: int) -> np.ndarray:
    if q_max < 1:
        raise ValueError("Q_MAX должен быть ≥ 1")
    if points < 1:
        raise ValueError("Q_POINTS должен быть ≥ 1")
    if mode == "linear":
        return np.linspace(1, q_max, points)
    if mode == "log":
        return np.logspace(np.log10(1), np.log10(q_max), points)
    raise ValueError("Q_MODE должен быть 'linear' или 'log'")


def load_bond_history(secid: str) -> pd.DataFrame:
    start = 0
    rows_all = []
    while True:
        url = (
            "https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/"
            f"{secid}.json"
        )
        response = request_get(url, params={"start": start, "iss.meta": "off"}, timeout=200)
        js = response.json()
        rows = js.get("history", {}).get("data", [])
        cols = js.get("history", {}).get("columns", [])
        if not rows:
            break
        rows_all.extend(rows)
        start += len(rows)
    return pd.DataFrame(rows_all, columns=cols)


def load_bond_yield_data(secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/bonds/"
        f"securities/{secid}/marketdata_yields.json"
    )
    response = request_get(url, params={"iss.meta": "off"}, timeout=200)
    js = response.json()
    if "marketdata_yields" not in js:
        raise ValueError("Нет блока marketdata_yields")
    df = pd.DataFrame(
        js["marketdata_yields"]["data"],
        columns=js["marketdata_yields"]["columns"],
    )
    for col in ["PRICE", "DURATIONWAPRICE", "EFFECTIVEYIELDWAPRICE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["PRICE", "DURATIONWAPRICE", "EFFECTIVEYIELDWAPRICE"])


def calculate_bond_delta_p(
    secid: str,
    c_value: float,
    date_from: str,
    date_to: str,
    q_mode: str,
    q_max: int,
    q_points: int = 50,
) -> tuple[pd.DataFrame, dict]:
    df_hist = load_bond_history(secid)
    df_yield = load_bond_yield_data(secid)
    if df_yield.empty:
        raise ValueError("Нет данных marketdata_yields для расчёта ΔP")

    df_hist["TRADEDATE"] = pd.to_datetime(df_hist["TRADEDATE"])
    df_hist = df_hist[
        (df_hist["TRADEDATE"] >= pd.to_datetime(date_from))
        & (df_hist["TRADEDATE"] <= pd.to_datetime(date_to))
    ]
    df_hist = df_hist[["TRADEDATE", "HIGH", "LOW", "CLOSE", "VALUE"]]
    df_hist[["HIGH", "LOW", "CLOSE", "VALUE"]] = df_hist[
        ["HIGH", "LOW", "CLOSE", "VALUE"]
    ].apply(pd.to_numeric, errors="coerce")
    df_hist = df_hist.dropna()
    if df_hist.empty:
        raise ValueError("Нет данных для расчёта σy")

    df_day = (
        df_hist.groupby("TRADEDATE", as_index=False)
        .agg({"HIGH": "mean", "LOW": "mean", "CLOSE": "mean", "VALUE": "sum"})
        .sort_values("TRADEDATE")
    )
    t_len = len(df_day)
    if t_len == 0:
        raise ValueError("Пустой период наблюдений")

    sigma_y = ((df_day["HIGH"] - df_day["LOW"]) / df_day["CLOSE"]).sum() / t_len
    mdtv = np.median(df_day["VALUE"])
    if not np.isfinite(sigma_y) or sigma_y <= 0:
        raise ValueError("Некорректный σy")
    if not np.isfinite(mdtv) or mdtv <= 0:
        raise ValueError(f"Некорректный MDTV = {mdtv}")

    q_vec = generate_q(q_mode, q_max, q_points)
    delta_y = c_value * sigma_y * np.sqrt(q_vec / mdtv)

    last = df_yield.iloc[-1]
    price = float(last["PRICE"])
    ytm = float(last["EFFECTIVEYIELDWAPRICE"]) / 100
    duration = float(last["DURATIONWAPRICE"]) / 364
    dmod = duration / (1 + ytm)

    delta_p = dmod * price * delta_y
    delta_p_pct = delta_p / price

    result = pd.DataFrame({"Q": q_vec, "DeltaP_pct": delta_p_pct})
    meta = {
        "ISIN": secid,
        "T": t_len,
        "SigmaY": float(sigma_y),
        "MDTV": float(mdtv),
        "Price": price,
        "YTM": ytm,
        "Dmod": dmod,
    }
    return result, meta


def parse_number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).strip().replace(" ", "").replace(",", ".")
        if cleaned == "":
            return None
        return float(cleaned)
    except Exception:
        return None


def extract_secid_shortname(row):
    if isinstance(row, dict):
        secid = row.get("SECID") or row.get("secid") or ""
        shortname = row.get("SHORTNAME") or row.get("shortname") or ""
        return secid, shortname
    if isinstance(row, (list, tuple)):
        secid = row[0] if len(row) > 0 else ""
        shortname = row[1] if len(row) > 1 else ""
        return secid, shortname
    return "", ""


@st.cache_data(ttl=3600)
def fetch_forts_securities():
    url = "https://iss.moex.com/iss/engines/futures/markets/forts/securities.xml"
    params = {
        "iss.meta": "off",
        "iss.only": "securities",
        "securities.columns": "SECID,SHORTNAME",
    }
    r = request_get(url, timeout=200, params=params)
    xml_content = r.content.decode("utf-8", errors="ignore")
    xml_content = re.sub(r'\sxmlns="[^"]+"', "", xml_content, count=1)
    root = ET.fromstring(xml_content)
    rows = []
    for el in root.iter():
        if el.tag.lower().endswith("row"):
            secid = el.attrib.get("SECID") or ""
            shortname = el.attrib.get("SHORTNAME") or ""
            if secid or shortname:
                rows.append([secid, shortname])
    return rows


@st.cache_data(ttl=3600)
def fetch_forts_secid_map():
    rows = fetch_forts_securities()
    mapping = {}
    for row in rows:
        secid, shortname = extract_secid_shortname(row)
        if shortname:
            mapping[str(shortname).upper()] = secid
    return mapping


def normalize_trade_key(value: str) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return normalized


def to_decimal(value) -> Decimal:
    return Decimal(str(value))


def money_decimal(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@st.cache_data(ttl=3600)
def get_usd_rub_cb_today():
    url = "https://www.cbr.ru/scripts/XML_daily.asp"
    r = requests.get(url, timeout=10)
    r.raise_for_status()

    root = ET.fromstring(r.content)
    for valute in root.findall("Valute"):
        char_code = valute.findtext("CharCode")
        if char_code == "USD":
            value_str = valute.findtext("Value", "").replace(",", ".")
            nominal_str = valute.findtext("Nominal", "1")
            usd_rub = Decimal(value_str) / Decimal(nominal_str)
            usd_rub = usd_rub.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            return {
                "date": root.attrib.get("Date"),
                "usd_rub": usd_rub,
            }

    raise RuntimeError("Не удалось получить курс USD/RUB с ЦБ")


def fetch_vm_data(trade_name: str, forts_rows=None):
    trade_name_clean = trade_name.strip()
    trade_name_upper = trade_name_clean.upper()
    trade_name_norm = normalize_trade_key(trade_name_clean)
    if forts_rows is None:
        forts_rows = fetch_forts_securities()
    secid_map = {}
    for row in forts_rows:
        secid, shortname = extract_secid_shortname(row)
        if shortname:
            secid_map[str(shortname).upper()] = secid
    secid = secid_map.get(trade_name_upper)
    if not secid and trade_name_norm:
        normalized_map = {
            normalize_trade_key(shortname): secid_value for shortname, secid_value in secid_map.items()
        }
        secid = normalized_map.get(trade_name_norm)
    if not secid:
        secid_match = [
            extract_secid_shortname(row)[0]
            for row in forts_rows
            if str(extract_secid_shortname(row)[0]).upper() == trade_name_upper
        ]
        if secid_match:
            secid = secid_match[0]
        else:
            partial = [
                extract_secid_shortname(row)[0]
                for row in forts_rows
                if trade_name_upper in str(extract_secid_shortname(row)[1]).upper()
            ]
            if len(partial) == 1:
                secid = partial[0]
            elif len(partial) > 1:
                raise RuntimeError(
                    f"Найдено несколько контрактов для {trade_name_clean}: {', '.join(partial[:5])}"
                )
            elif trade_name_norm:
                normalized_matches = [
                    extract_secid_shortname(row)[0]
                    for row in forts_rows
                    if trade_name_norm in normalize_trade_key(extract_secid_shortname(row)[1])
                ]
                if len(normalized_matches) == 1:
                    secid = normalized_matches[0]
                elif len(normalized_matches) > 1:
                    raise RuntimeError(
                        f"Найдено несколько контрактов для {trade_name_clean}: {', '.join(normalized_matches[:5])}"
                    )
    if not secid:
        raise RuntimeError(f"Контракт {trade_name_clean} не найден в FORTS")

    spec_url = f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{secid}.json"
    spec_params = {
        "iss.meta": "off",
        "iss.only": "securities,marketdata",
        "securities.columns": "PREVSETTLEPRICE,MINSTEP,STEPPRICE,LASTSETTLEPRICE",
        "marketdata.columns": "LAST,UPDATETIME",
    }
    spec = request_get(spec_url, timeout=2000, params=spec_params).json()

    sec_columns = spec.get("securities", {}).get("columns", [])
    sec_data = spec.get("securities", {}).get("data", [])
    if not sec_data:
        raise RuntimeError("Не удалось получить спецификацию контракта FORTS")
    sec_row = dict(zip(sec_columns, sec_data[0]))

    prev_settle_raw = sec_row.get("PREVSETTLEPRICE")
    minstep_raw = sec_row.get("MINSTEP")
    stepprice_raw = sec_row.get("STEPPRICE")
    last_settle_raw = sec_row.get("LASTSETTLEPRICE")

    market_columns = spec.get("marketdata", {}).get("columns", [])
    market_data = spec.get("marketdata", {}).get("data", [])
    market_row = dict(zip(market_columns, market_data[0])) if market_data else {}
    last_raw = market_row.get("LAST")
    update_time_raw = market_row.get("UPDATETIME")

    if prev_settle_raw is None or minstep_raw is None or stepprice_raw is None:
        raise RuntimeError("В спецификации FORTS отсутствуют обязательные поля цены")

    prev_settle = to_decimal(prev_settle_raw)
    minstep = to_decimal(minstep_raw)
    stepprice = to_decimal(stepprice_raw)
    last_settle = to_decimal(last_settle_raw) if last_settle_raw is not None else None
    last_price = to_decimal(last_raw) if last_raw is not None else None

    hist_url = f"https://iss.moex.com/iss/history/engines/futures/markets/forts/securities/{secid}.json"
    hist_params = {
        "iss.meta": "off",
        "iss.only": "history",
        "history.columns": "TRADEDATE,SETTLEPRICEDAY",
        "sort_order": "desc",
        "limit": 1,
    }
    history = request_get(hist_url, timeout=2000, params=hist_params).json()
    rows = history.get("history", {}).get("data", [])
    if not rows or rows[0][1] is None:
        raise RuntimeError("Дневной клиринг ещё не опубликован")

    trade_date, day_settle_raw = rows[0]
    day_settle = to_decimal(day_settle_raw)

    multiplier = stepprice / minstep
    vm_clearing = (day_settle - prev_settle) * multiplier
    ref_price = last_price if last_price is not None else (last_settle if last_settle is not None else day_settle)
    vm_live = (ref_price - prev_settle) * multiplier

    return {
        "TRADE_NAME": trade_name_clean,
        "SECID": secid,
        "TRADEDATE": trade_date,
        "PREV_PRICE": float(money_decimal(prev_settle)),
        "LAST_SETTLE_PRICE": float(money_decimal(last_settle)) if last_settle is not None else None,
        "TODAY_PRICE": float(money_decimal(day_settle)),
        "LAST_PRICE": float(money_decimal(last_price)) if last_price is not None else None,
        "QUOTE_TIME": str(update_time_raw) if update_time_raw is not None else None,
        "MULTIPLIER": float(multiplier),
        "VM_CLEARING": float(money_decimal(vm_clearing)),
        "VM": float(money_decimal(vm_live)),
    }


# ---------------------------
# Safe CSV/Excel reading helpers
# ---------------------------
def safe_read_csv_string(content: str) -> pd.DataFrame:
    content = content.replace("\r\n", "\n").strip()
    sample = content[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        sep = dialect.delimiter
    except Exception:
        sep = ","
    try:
        df = pd.read_csv(StringIO(content), sep=sep, dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def safe_read_filelike(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    try:
        if name.lower().endswith(".csv"):
            raw = uploaded_file.getvalue().decode("utf-8-sig")
            return safe_read_csv_string(raw)
        return pd.read_excel(uploaded_file, dtype=str)
    except Exception:
        return pd.DataFrame()


# ---------------------------
# ISIN validation (format + checksum)
# ---------------------------
def isin_format_valid(isin: str) -> bool:
    if not isin or not isinstance(isin, str):
        return False
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", isin.strip().upper()))


def isin_checksum_valid(isin: str) -> bool:
    """ISIN checksum (Luhn-like). Returns True for valid ISINs."""
    if not isin_format_valid(isin):
        return False
    s = isin.strip().upper()
    converted = ""
    for ch in s[:-1]:
        if ch.isdigit():
            converted += ch
        else:
            converted += str(ord(ch) - 55)
    digits = converted + s[-1]
    arr = [int(x) for x in digits]
    total = 0
    parity = len(arr) % 2
    for i, d in enumerate(arr):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------
# Load TQOB/TQCB board XML caches (for fallback)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_board_xml(board: str):
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/bonds/boards/"
        f"{board.lower()}/securities.xml?marketprice_board=3&iss.meta=off"
    )
    try:
        r = request_get(url, timeout=200)
        xml_content = r.content.decode("utf-8", errors="ignore")
        xml_content = re.sub(r'\sxmlns="[^"]+"', "", xml_content, count=1)
        root = ET.fromstring(xml_content)
        mapping = {}
        for el in root.iter():
            if el.tag.lower().endswith("row"):
                attrs = {k.upper(): v for k, v in el.attrib.items()}
                isin = attrs.get("ISIN", "").strip().upper()
                secid = attrs.get("SECID", "").strip().upper()
                emitterid = attrs.get("EMITTERID", "").strip()
                if isin:
                    mapping[isin] = {"SECID": secid or None, "EMITTERID": emitterid or None, **attrs}
        return mapping
    except Exception:
        return {}


TQOB_MAP = fetch_board_xml("tqob")
TQCB_MAP = fetch_board_xml("tqcb")

# ---------------------------
# Fetch emitter & secid (with caching)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_emitter_and_secid(isin: str):
    isin = str(isin).strip().upper()
    if not isin:
        return None, None
    emitter_id = None
    secid = None

    try:
        url = f"https://iss.moex.com/iss/securities/{isin}.json"
        r = request_get(url, timeout=10)
        data = r.json()
        securities = data.get("securities", {})
        cols = securities.get("columns", [])
        rows = securities.get("data", [])
        if rows and cols:
            first = rows[0]
            col_map = {c.upper(): i for i, c in enumerate(cols)}
            if "EMITTER_ID" in col_map:
                emitter_id = first[col_map["EMITTER_ID"]]
            elif "EMITTERID" in col_map:
                emitter_id = first[col_map["EMITTERID"]]
            if "SECID" in col_map:
                secid = first[col_map["SECID"]]
    except Exception:
        pass

    if not secid:
        try:
            url = f"https://iss.moex.com/iss/securities/{isin}.xml?iss.meta=off"
            r = request_get(url, timeout=10)
            xml_content = r.content.decode("utf-8", errors="ignore")
            xml_content = re.sub(r'\sxmlns="[^"]+"', "", xml_content, count=1)
            root = ET.fromstring(xml_content)
            for node in root.iter():
                name_attr = (node.attrib.get("name") or "").upper()
                val_attr = node.attrib.get("value") or ""
                if name_attr == "SECID":
                    secid = val_attr
                elif name_attr in ("EMITTER_ID", "EMITTERID"):
                    emitter_id = val_attr
        except Exception:
            pass

    if not secid:
        mapping = TQOB_MAP.get(isin) or TQCB_MAP.get(isin)
        if mapping:
            secid = mapping.get("SECID")
            if not emitter_id:
                emitter_id = mapping.get("EMITTERID")

    return emitter_id, secid


# ---------------------------
# Core: get bond data per ISIN
# ---------------------------
@st.cache_data(ttl=3600)
def get_bond_data(isin: str):
    isin = str(isin).strip().upper()
    try:
        emitter_id, secid = fetch_emitter_and_secid(isin)
        secname = maturity_date = put_date = call_date = None
        record_date = coupon_date = None
        coupon_currency = None
        coupon_value = None
        coupon_value_rub = None
        coupon_value_prc = None

        if secid:
            try:
                url_info = f"https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{secid}.json"
                r = request_get(url_info, timeout=10)
                data_info = r.json()
                rows_info = data_info.get("securities", {}).get("data", [])
                cols_info = data_info.get("securities", {}).get("columns", [])
                if rows_info and cols_info:
                    info = dict(zip([c.upper() for c in cols_info], rows_info[0]))
                    secname = info.get("SECNAME") or info.get("SEC_NAME") or secname
                    maturity_date = info.get("MATDATE") or maturity_date
                    put_date = info.get("PUTOPTIONDATE") or put_date
                    call_date = info.get("CALLOPTIONDATE") or call_date
            except Exception:
                pass

        if not secname or not maturity_date:
            try:
                url_info_isin = f"https://iss.moex.com/iss/securities/{isin}.json"
                r = request_get(url_info_isin, timeout=10)
                data_info_isin = r.json()
                rows = data_info_isin.get("securities", {}).get("data", [])
                cols = data_info_isin.get("securities", {}).get("columns", [])
                if rows and cols:
                    info = dict(zip([c.upper() for c in cols], rows[0]))
                    secname = secname or info.get("SECNAME") or info.get("SEC_NAME")
                    maturity_date = maturity_date or info.get("MATDATE") or info.get("MATDATE")
                    put_date = put_date or info.get("PUTOPTIONDATE") or info.get("PUT_OPTION_DATE")
                    call_date = call_date or info.get("CALLOPTIONDATE") or info.get("CALL_OPTION_DATE")
            except Exception:
                pass

        coupons_data = None
        columns_coupons = []
        bondization_faceunit = None

        def try_fetch_bondization(identifier: str):
            try:
                url_coupons = (
                    "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/"
                    f"bondization/{identifier}.json?iss.only=coupons,bondization&iss.meta=off"
                )
                r = request_get(url_coupons, timeout=10)
                data = r.json()
                coupons = data.get("coupons", {}).get("data", [])
                cols = data.get("coupons", {}).get("columns", [])
                bond_rows = data.get("bondization", {}).get("data", [])
                bond_cols = data.get("bondization", {}).get("columns", [])
                faceunit = None
                if bond_rows and bond_cols:
                    bond_info = dict(zip([c.upper() for c in bond_cols], bond_rows[0]))
                    faceunit = bond_info.get("FACEUNIT") or bond_info.get("FACEUNIT_S")
                return coupons, cols, faceunit
            except Exception:
                return None, [], None

        if secid:
            coupons_data, columns_coupons, bondization_faceunit = try_fetch_bondization(secid)

        if isin and (not coupons_data or not columns_coupons):
            coupons_data_fallback, columns_coupons_fallback, bondization_faceunit_fallback = try_fetch_bondization(isin)
            if coupons_data_fallback and columns_coupons_fallback:
                coupons_data = coupons_data_fallback
                columns_coupons = columns_coupons_fallback
            if not bondization_faceunit:
                bondization_faceunit = bondization_faceunit_fallback or bondization_faceunit

        if coupons_data and columns_coupons:
            df_coupons = pd.DataFrame(coupons_data, columns=columns_coupons)
            cols_upper = [c.upper() for c in df_coupons.columns]
            df_coupons.columns = cols_upper
            today = pd.to_datetime(datetime.today().date())
            possible_coupon_date_cols = [c for c in cols_upper if "COUPON" in c and "DATE" in c]
            possible_record_date_cols = [c for c in cols_upper if "RECORD" in c and "DATE" in c]

            def next_future_date(series):
                try:
                    s = pd.to_datetime(series, errors="coerce")
                    s = s[s >= today]
                    if not s.empty:
                        return s.min().strftime("%Y-%m-%d")
                except Exception:
                    pass
                return None

            coupon_found = None
            for col in possible_coupon_date_cols:
                candidate = next_future_date(df_coupons[col])
                if candidate:
                    coupon_found = candidate
                    break
            if not coupon_found:
                all_dates = []
                for col in df_coupons.columns:
                    try:
                        s = pd.to_datetime(df_coupons[col], errors="coerce")
                        s = s[s >= today]
                        if not s.empty:
                            all_dates.append(s.min())
                    except Exception:
                        pass
                if all_dates:
                    coupon_found = min(all_dates).strftime("%Y-%m-%d")
            coupon_date = coupon_found

            record_found = None
            for col in possible_record_date_cols:
                candidate = next_future_date(df_coupons[col])
                if candidate:
                    record_found = candidate
                    break
            record_date = record_found

            if bondization_faceunit:
                coupon_currency = bondization_faceunit
            else:
                faceunit_cols = [c for c in df_coupons.columns if "FACEUNIT" in c or c == "FACEUNIT_S"]
                if faceunit_cols:
                    for c in faceunit_cols:
                        vals = df_coupons[c].dropna().astype(str)
                        if not vals.empty and vals.iloc[0].strip():
                            coupon_currency = vals.iloc[0].strip()
                            break

            val_col = None
            val_rub_col = None
            val_prc_col = None
            for c in df_coupons.columns:
                uc = c.upper()
                if uc in ("VALUE", "VALUE_COUPON", "COUPONVALUE") and not val_col:
                    val_col = c
                if "VALUE_RUB" in uc and not val_rub_col:
                    val_rub_col = c
                if uc in ("VALUEPRC", "VALUE_PRC", "VALUE%") and not val_prc_col:
                    val_prc_col = c
            if not val_col:
                for c in df_coupons.columns:
                    if re.match(r"^VALUE$", c, flags=re.IGNORECASE):
                        val_col = c
                        break
            if not val_rub_col:
                for c in df_coupons.columns:
                    if re.search(r"RUB", c, flags=re.IGNORECASE):
                        if "VALUE" in c.upper() or "RUB" in c.upper():
                            val_rub_col = c
                            break
            if not val_prc_col:
                for c in df_coupons.columns:
                    if re.search(r"PRC|PERC|%|PERCENT", c, flags=re.IGNORECASE):
                        val_prc_col = c
                        break

            chosen_row = None
            if coupon_date:
                for c in possible_coupon_date_cols:
                    try:
                        mask = pd.to_datetime(df_coupons[c], errors="coerce").dt.strftime("%Y-%m-%d") == coupon_date
                        rows = df_coupons[mask]
                        if not rows.empty:
                            chosen_row = rows.iloc[0]
                            break
                    except Exception:
                        pass
            if chosen_row is None:
                for idx in range(len(df_coupons)):
                    row = df_coupons.iloc[idx]
                    has_val = False
                    for colcheck in (val_col, val_rub_col, val_prc_col):
                        try:
                            if colcheck and pd.notnull(row.get(colcheck)):
                                has_val = True
                                break
                        except Exception:
                            pass
                    if has_val:
                        chosen_row = row
                        break

            def norm_str(v):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return None
                try:
                    return str(v)
                except Exception:
                    return None

            if chosen_row is not None:
                coupon_value = norm_str(chosen_row.get(val_col)) if val_col else None
                coupon_value_rub = norm_str(chosen_row.get(val_rub_col)) if val_rub_col else None
                coupon_value_prc = norm_str(chosen_row.get(val_prc_col)) if val_prc_col else None

        if (not record_date or not coupon_date or not coupon_currency) and (TQOB_MAP or TQCB_MAP):
            mapping = TQOB_MAP.get(isin) or TQCB_MAP.get(isin)
            if mapping:
                if not record_date:
                    record_date = mapping.get("RECORDDATE") or mapping.get("RECORD_DATE") or mapping.get("RECORD")
                if not coupon_date:
                    coupon_date = mapping.get("COUPONDATE") or mapping.get("COUPON_DATE") or mapping.get("COUPON")
                if not coupon_currency:
                    coupon_currency = mapping.get("FACEUNIT") or mapping.get("FACEUNIT_S") or mapping.get("FACEUNIT")

        def fmt(date):
            if pd.isna(date) or not date:
                return None
            try:
                return pd.to_datetime(date).strftime("%Y-%m-%d")
            except Exception:
                return None

        return {
            "ISIN": isin,
            "Код эмитента": emitter_id or "",
            "Наименование инструмента": secname or "",
            "Дата погашения": fmt(maturity_date),
            "Дата оферты Put": fmt(put_date),
            "Дата оферты Call": fmt(call_date),
            "Дата фиксации купона": fmt(record_date),
            "Дата купона": fmt(coupon_date),
            "Валюта купона": coupon_currency or "",
            "Купон в валюте": coupon_value or "",
            "Купон в Руб": coupon_value_rub or "",
            "Купон %": coupon_value_prc or "",
        }
    except Exception:
        return {
            "ISIN": isin,
            "Код эмитента": "",
            "Наименование инструмента": "",
            "Дата погашения": None,
            "Дата оферты Put": None,
            "Дата оферты Call": None,
            "Дата фиксации купона": None,
            "Дата купона": None,
            "Валюта купона": "",
            "Купон в валюте": "",
            "Купон в Руб": "",
            "Купон %": "",
        }


# ---------------------------
# Calendar helpers
# ---------------------------
@st.cache_data(ttl=3600)
def get_bond_schedule(isin: str):
    isin = str(isin).strip().upper()
    emitter_id, secid = fetch_emitter_and_secid(isin)
    maturity_date = put_date = call_date = None
    coupon_events = {}
    facevalue = None

    if secid:
        try:
            url_info = f"https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{secid}.json"
            r = request_get(url_info, timeout=10)
            data_info = r.json()
            rows_info = data_info.get("securities", {}).get("data", [])
            cols_info = data_info.get("securities", {}).get("columns", [])
            if rows_info and cols_info:
                info = dict(zip([c.upper() for c in cols_info], rows_info[0]))
                maturity_date = info.get("MATDATE") or maturity_date
                put_date = info.get("PUTOPTIONDATE") or put_date
                call_date = info.get("CALLOPTIONDATE") or call_date
                if facevalue is None:
                    facevalue = parse_number(info.get("FACEVALUE") or info.get("FACEVALUE_RUB"))
        except Exception:
            pass

    if not maturity_date:
        try:
            url_info_isin = f"https://iss.moex.com/iss/securities/{isin}.json"
            r = request_get(url_info_isin, timeout=10)
            data_info_isin = r.json()
            rows = data_info_isin.get("securities", {}).get("data", [])
            cols = data_info_isin.get("securities", {}).get("columns", [])
            if rows and cols:
                info = dict(zip([c.upper() for c in cols], rows[0]))
                maturity_date = maturity_date or info.get("MATDATE")
                put_date = put_date or info.get("PUTOPTIONDATE") or info.get("PUT_OPTION_DATE")
                call_date = call_date or info.get("CALLOPTIONDATE") or info.get("CALL_OPTION_DATE")
                if facevalue is None:
                    facevalue = parse_number(info.get("FACEVALUE") or info.get("FACEVALUE_RUB"))
        except Exception:
            pass

    def try_fetch_bondization(identifier: str):
        try:
            url_coupons = (
                "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/"
                f"bondization/{identifier}.json?iss.only=coupons,bondization&iss.meta=off"
            )
            r = request_get(url_coupons, timeout=10)
            data = r.json()
            coupons = data.get("coupons", {}).get("data", [])
            cols = data.get("coupons", {}).get("columns", [])
            bond_rows = data.get("bondization", {}).get("data", [])
            bond_cols = data.get("bondization", {}).get("columns", [])
            bond_info = None
            if bond_rows and bond_cols:
                bond_info = dict(zip([c.upper() for c in bond_cols], bond_rows[0]))
            return coupons, cols, bond_info
        except Exception:
            return None, [], None

    coupons_data = columns_coupons = None
    bond_info = None
    if secid:
        coupons_data, columns_coupons, bond_info = try_fetch_bondization(secid)

    if isin and (not coupons_data or not columns_coupons):
        coupons_data_fallback, columns_coupons_fallback, bond_info_fallback = try_fetch_bondization(isin)
        if coupons_data_fallback and columns_coupons_fallback:
            coupons_data = coupons_data_fallback
            columns_coupons = columns_coupons_fallback
        if not bond_info:
            bond_info = bond_info_fallback or bond_info

    if bond_info:
        for key, value in bond_info.items():
            if key.startswith("FACEVALUE") and value not in (None, "", "None"):
                facevalue = parse_number(value)
                if facevalue is not None:
                    break

    if coupons_data and columns_coupons:
        df_coupons = pd.DataFrame(coupons_data, columns=columns_coupons)
        df_coupons.columns = [c.upper() for c in df_coupons.columns]
        date_cols = [c for c in df_coupons.columns if "COUPON" in c and "DATE" in c]
        if not date_cols:
            date_cols = [c for c in df_coupons.columns if c.endswith("DATE")]

        val_rub_col = next((c for c in df_coupons.columns if "VALUE_RUB" in c), None)
        val_col = next(
            (c for c in df_coupons.columns if c in ("VALUE", "VALUE_COUPON", "COUPONVALUE")),
            None,
        )
        for _, row in df_coupons.iterrows():
            coupon_date = None
            for col in date_cols:
                dt = pd.to_datetime(row.get(col), errors="coerce")
                if pd.notna(dt):
                    coupon_date = dt.strftime("%Y-%m-%d")
                    break
            if not coupon_date:
                continue
            raw_value = row.get(val_rub_col) if val_rub_col else row.get(val_col)
            coupon_value = parse_number(raw_value)
            coupon_events[coupon_date] = coupon_value

    def fmt(date):
        if pd.isna(date) or not date:
            return None
        try:
            return pd.to_datetime(date).strftime("%Y-%m-%d")
        except Exception:
            return None

    return {
        "ISIN": isin,
        "Код эмитента": emitter_id or "",
        "Дата погашения": fmt(maturity_date),
        "Дата оферты Put": fmt(put_date),
        "Дата оферты Call": fmt(call_date),
        "Купоны": coupon_events,
        "Номинал": facevalue,
    }


# ---------------------------
# Sequential fetch with safe progress updates
# ---------------------------
def fetch_isins(isins, show_progress=True):
    results = []
    total = len(isins)
    if total == 0:
        return results

    progress_bar = None
    progress_text = None
    if show_progress:
        try:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        except Exception:
            progress_bar = None
            progress_text = None

    for idx, isin in enumerate(isins, start=1):
        try:
            data = get_bond_data(isin)
        except Exception:
            data = {
                "ISIN": isin,
                "Код эмитента": "",
                "Наименование инструмента": "",
                "Дата погашения": None,
                "Дата оферты Put": None,
                "Дата оферты Call": None,
                "Дата фиксации купона": None,
                "Дата купона": None,
                "Валюта купона": "",
                "Купон в валюте": "",
                "Купон в Руб": "",
                "Купон %": "",
            }
        results.append(data)
        if progress_bar:
            try:
                progress_bar.progress(idx / total)
            except Exception:
                pass
        if progress_text:
            try:
                progress_text.text(f"Обработано {idx}/{total} ISIN")
            except Exception:
                pass
    try:
        time.sleep(0.12)
    except Exception:
        pass
    return results


# ---------------------------
# Calendar view
# ---------------------------
if st.session_state["active_view"] == "calendar":
    st.subheader("📅 Календарь выплат")
    st.markdown(
        "Введите список бумаг и их количество, чтобы построить календарь выплат "
        "(купоны, погашения, оферты) по вашему портфелю."
    )
    st.markdown("**Ручной ввод:** `ISIN | Amount` (количество бумаг).")
    calendar_input = st.text_area(
        "Введите список бумаг",
        height=160,
        placeholder="RU000A0JX0J2 | 100\nRU000A0ZZZY1 | 50",
        key="calendar_manual_input",
    )
    if st.button("Построить календарь", key="build_calendar"):
        raw_lines = [line.strip() for line in calendar_input.splitlines() if line.strip()]
        entries = []
        invalid_isins = []
        for line in raw_lines:
            parts = [p.strip() for p in re.split(r"[,.;|/\t]+", line) if p.strip()]
            if not parts:
                continue
            isin = parts[0].upper()
            amount = parse_number(parts[1]) if len(parts) > 1 else 1.0
            if amount is None or amount <= 0:
                amount = 1.0
            if not isin_format_valid(isin) or not isin_checksum_valid(isin):
                invalid_isins.append(isin)
                continue
            entries.append({"ISIN": isin, "Amount": amount})

        if invalid_isins:
            st.warning(
                "Некорректные ISIN пропущены: "
                f"{', '.join(invalid_isins[:10])}{'...' if len(invalid_isins) > 10 else ''}"
            )
        if not entries:
            st.error("Нет валидных ISIN для построения календаря.")
        else:
            timeline_data = {}
            all_dates = set()
            with st.spinner("Загружаем расписание выплат..."):
                for entry in entries:
                    isin = entry["ISIN"]
                    amount = entry["Amount"]
                    try:
                        schedule = get_bond_schedule(isin)
                    except Exception:
                        schedule = {}
                    row = {}
                    today = datetime.today().date()
                    maturity_date = None
                    for key in ("Дата погашения", "Дата оферты Put", "Дата оферты Call"):
                        event_date = schedule.get(key)
                        if event_date:
                            try:
                                parsed_date = pd.to_datetime(event_date).date()
                            except Exception:
                                parsed_date = None
                            if parsed_date and (maturity_date is None or parsed_date < maturity_date):
                                maturity_date = parsed_date
                            if parsed_date and parsed_date >= today:
                                all_dates.add(event_date)

                    for date, value in schedule.get("Купоны", {}).items():
                        try:
                            coupon_date = pd.to_datetime(date).date()
                        except Exception:
                            continue
                        if coupon_date < today:
                            continue
                        if maturity_date and coupon_date > maturity_date:
                            continue
                        all_dates.add(date)
                        if value is None:
                            continue
                        scaled = value * amount
                        row[date] = row.get(date, 0) + scaled

                    facevalue = schedule.get("Номинал")
                    for key in ("Дата погашения", "Дата оферты Put", "Дата оферты Call"):
                        event_date = schedule.get(key)
                        if event_date and facevalue is not None:
                            try:
                                parsed_date = pd.to_datetime(event_date).date()
                            except Exception:
                                parsed_date = None
                            if parsed_date and parsed_date >= today:
                                row[event_date] = row.get(event_date, 0) + facevalue * amount
                    timeline_data[isin] = row

            sorted_dates = sorted(all_dates)
            df_timeline = pd.DataFrame(index=[e["ISIN"] for e in entries], columns=sorted_dates, dtype=float)
            for isin, row in timeline_data.items():
                for date, value in row.items():
                    df_timeline.loc[isin, date] = value

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_timeline.to_excel(writer, sheet_name="calendar")
            st.session_state["calendar_last_report"] = {
                "df": df_timeline,
                "xlsx": output.getvalue(),
                "csv": df_timeline.to_csv(index=True).encode("utf-8-sig"),
            }

    calendar_report = st.session_state.get("calendar_last_report")
    if calendar_report:
        st.dataframe(calendar_report["df"], use_container_width=True)
        st.download_button(
            label="💾 Скачать календарь (Excel)",
            data=calendar_report["xlsx"],
            file_name="bond_calendar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="calendar_xlsx_dl",
        )
        st.download_button(
            label="💾 Скачать календарь (CSV)",
            data=calendar_report["csv"],
            file_name="bond_calendar.csv",
            mime="text/csv",
            key="calendar_csv_dl",
        )
        render_email_compose_section(
            "Календарь выплат",
            "calendar_report",
            "bond_calendar.xlsx",
            calendar_report["xlsx"],
        )
    st.stop()

# ---------------------------
# VM view
# ---------------------------
if st.session_state["active_view"] == "vm":
    st.subheader("🧮 Расчет VM")
    if "forts_contracts" not in st.session_state:
        with st.spinner("Загружаем список контрактов FORTS..."):
            try:
                st.session_state["forts_contracts"] = fetch_forts_securities()
            except Exception:
                st.session_state["forts_contracts"] = []
                st.warning("Не удалось загрузить список контрактов FORTS.")
    trade_name = st.text_input("TRADE_NAME (SHORTNAME биржи)", value="", key="vm_trade_name")
    quantity = st.number_input(
        "Кол-во (целое, неотрицательное)",
        min_value=0,
        step=1,
        value=0,
        format="%d",
        key="vm_quantity",
    )
    if st.button("Рассчитать VM", key="vm_calculate"):
        if not trade_name.strip():
            st.error("Введите TRADE_NAME.")
        else:
            try:
                vm_data = fetch_vm_data(trade_name.strip(), st.session_state.get("forts_contracts"))
                position_vm = vm_data["VM"] * quantity
                usd_rub_data = get_usd_rub_cb_today()
                usd_rub = float(usd_rub_data["usd_rub"])
                price_for_limit = vm_data["LAST_PRICE"] if vm_data.get("LAST_PRICE") is not None else vm_data["TODAY_PRICE"]
                limit_sum = (0.05 * price_for_limit * quantity * usd_rub) + (max(0, position_vm))
                st.session_state["vm_last_report"] = {
                    "TRADE_NAME": vm_data["TRADE_NAME"],
                    "SECID": vm_data["SECID"],
                    "TRADEDATE": vm_data["TRADEDATE"],
                    "LAST_SETTLE_PRICE": vm_data["LAST_SETTLE_PRICE"],
                    "TODAY_PRICE": vm_data["TODAY_PRICE"],
                    "LAST_PRICE": vm_data.get("LAST_PRICE"),
                    "QUOTE_TIME": vm_data.get("QUOTE_TIME"),
                    "MULTIPLIER": vm_data["MULTIPLIER"],
                    "VM": vm_data["VM"],
                    "VM_CLEARING": vm_data["VM_CLEARING"],
                    "QUANTITY": quantity,
                    "POSITION_VM": position_vm,
                    "USD_RUB": usd_rub_data["usd_rub"],
                    "USD_RUB_DATE": usd_rub_data["date"],
                    "LIMIT_SUM": limit_sum,
                }
            except Exception as exc:
                st.error(str(exc))

    vm_report = st.session_state.get("vm_last_report")
    if vm_report:
        st.markdown(f"**Инструмент:** {vm_report['TRADE_NAME']}")
        st.markdown(f"**SECID:** {vm_report['SECID']}")
        st.markdown(f"**Дата клиринга:** {vm_report['TRADEDATE']}")
        st.markdown(f"**Расчетная цена последнего клиринга:** {vm_report['LAST_SETTLE_PRICE']}")
        st.markdown(f"**Последняя цена:** {vm_report.get('LAST_PRICE') if vm_report.get('LAST_PRICE') is not None else vm_report['TODAY_PRICE']}")
        st.markdown(f"**Время котировки:** {vm_report.get('QUOTE_TIME') or 'н/д'}")
        st.markdown(f"**Multiplier:** {vm_report['MULTIPLIER']}")
        st.markdown(f"**Вариационная маржа (по последней цене):** {vm_report['VM']:.2f}")
        st.markdown(f"**VM клиринговая (SETTLEPRICEDAY - PREVSETTLEPRICE):** {vm_report.get('VM_CLEARING', vm_report['VM']):.2f}")
        st.markdown(f"**Маржа позиции (VM × Кол-во):** {vm_report['POSITION_VM']:.2f}")
        st.markdown(f"**Сумма ограничения:** {vm_report['LIMIT_SUM']:.2f}")
        st.caption(f"USD/RUB: {vm_report['USD_RUB']} на {vm_report['USD_RUB_DATE']}")

        vm_df = pd.DataFrame([vm_report])
        vm_xlsx = ss.dataframe_to_excel_bytes(vm_df, sheet_name="vm_report")
        st.download_button(
            label="💾 Скачать VM (Excel)",
            data=vm_xlsx,
            file_name="vm_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="vm_report_xlsx_dl",
        )
        st.download_button(
            label="💾 Скачать VM (CSV)",
            data=vm_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="vm_report.csv",
            mime="text/csv",
            key="vm_report_csv_dl",
        )

        vm_mail_body = (
            "Коллеги, добрый день!\n\n"
            "Направляю отчёт по вариационной марже (VM).\n\n"
            f"Инструмент: {vm_report['TRADE_NAME']} ({vm_report['SECID']})\n"
            f"Дата клиринга: {vm_report['TRADEDATE']}\n"
            f"Кол-во: {vm_report['QUANTITY']}\n"
            f"VM (по последней цене): {vm_report['VM']:.2f}\n"
            f"VM клиринговая: {vm_report.get('VM_CLEARING', vm_report['VM']):.2f}\n"
            f"Маржа позиции: {vm_report['POSITION_VM']:.2f}\n"
            f"Сумма ограничения: {vm_report['LIMIT_SUM']:.2f}\n"
            f"USD/RUB: {vm_report['USD_RUB']} на {vm_report['USD_RUB_DATE']}\n\n"
            "Детали во вложении."
        )

        st.session_state["vm_report_default_body"] = vm_mail_body
        render_email_compose_section(
            "VM отчёт",
            "vm_report",
            "vm_report.xlsx",
            vm_xlsx,
        )

    st.stop()

# ---------------------------
# Sell_stres view
# ---------------------------
if st.session_state["active_view"] == "sell_stres":
    st.subheader("🧩 Sell_stres")

    init_sell_stres_state()

    share_tab, bond_tab = st.tabs(["Share", "Bond"])

    with share_tab:
        st.markdown("### Share")
        use_q_from_list = st.checkbox(
            "Вводить Q для каждого ISIN/Ticker (формат: ISIN/Ticker | Q)", value=False, key="share_q_per_isin"
        )
        if use_q_from_list:
            isin_q_input = st.text_area(
                "Введите ISIN или Ticker и Q (каждая строка: ISIN/Ticker | Q)",
                height=160,
                placeholder="RU0009029540 | 33000000000\nSBER | 25000000000",
                key="share_isin_q_input",
            )
        else:
            isin_input = st.text_area(
                "Введите или вставьте ISIN/Ticker (через Ctrl+V, пробел или запятую)",
                height=160,
                placeholder="RU0009029540\nSBER",
                key="share_isin_input",
            )

        c_value = st.number_input(
            "C (коэффициент, 0–1)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="share_c_value",
        )
        data_from = st.date_input(
            "Дата начала (data_from)",
            value=datetime(2024, 1, 1).date(),
            key="share_date_from",
        )
        q_max = st.number_input(
            "Q (максимум для построения вектора)",
            min_value=1,
            value=33_000_000_000,
            step=1_000_000,
            format="%d",
            key="share_q_max",
            disabled=use_q_from_list,
        )
        use_log = st.checkbox("Логарифмическое приближение", value=True, key="share_q_log")
        q_mode = "log" if use_log else "linear"

        if st.button("Рассчитать Sell_stres (Share)", key="share_calculate"):
            entries = []
            unresolved_identifiers = []
            if use_q_from_list:
                raw_lines = [line.strip() for line in isin_q_input.splitlines() if line.strip()]
                for line in raw_lines:
                    parts = [p.strip() for p in re.split(r"[|;	,]+", line) if p.strip()]
                    if not parts:
                        continue
                    identifier = parts[0].upper()
                    q_val = parse_number(parts[1]) if len(parts) > 1 else None
                    resolved_isin = resolve_share_identifier_to_isin(identifier)
                    if not resolved_isin:
                        unresolved_identifiers.append(identifier)
                        continue
                    if q_val is None or q_val <= 0:
                        st.warning(f"Некорректный Q для {identifier}: {parts[1] if len(parts) > 1 else ''}")
                        continue
                    entries.append({"ISIN": resolved_isin, "Q_MAX": int(q_val)})
            else:
                raw_text = isin_input.strip()
                if raw_text:
                    identifiers = re.split(r"[\s,;]+", raw_text)
                    identifiers = [i.strip().upper() for i in identifiers if i.strip()]
                    for identifier in identifiers:
                        resolved_isin = resolve_share_identifier_to_isin(identifier)
                        if not resolved_isin:
                            unresolved_identifiers.append(identifier)
                            continue
                        entries.append({"ISIN": resolved_isin, "Q_MAX": int(q_max)})

            if unresolved_identifiers:
                st.warning(
                    "Не удалось распознать ISIN/Ticker, записи пропущены: "
                    f"{', '.join(unresolved_identifiers[:10])}{'...' if len(unresolved_identifiers) > 10 else ''}"
                )

            if not entries:
                st.error("Нет валидных ISIN для обработки.")
            elif len(entries) > ss.MAX_SECURITIES_PER_RUN:
                st.error(
                    f"Слишком много бумаг за один запуск: {len(entries)}. "
                    f"Лимит: {ss.MAX_SECURITIES_PER_RUN}."
                )
            else:
                meta_rows = []
                results = {}
                progress_bar = st.progress(0.0)
                with st.spinner("Рассчитываем Sell_stres..."):
                    for idx, entry in enumerate(entries, start=1):
                        isin = entry["ISIN"]
                        try:
                            q_vector = ss.build_q_vector(q_mode, entry["Q_MAX"])
                            delta_df, meta = ss.calculate_share_delta_p(
                                request_get=request_get,
                                isin_to_secid=isin_to_secid,
                                isin=isin,
                                c_value=float(c_value),
                                date_from=data_from.strftime("%Y-%m-%d"),
                                q_values=q_vector,
                            )
                            results[isin] = delta_df
                            meta_rows.append(meta)
                        except Exception as exc:
                            st.error(f"{isin}: {exc}")
                        progress_bar.progress(idx / len(entries))

                show_tables = len(entries) == 1 and not use_q_from_list
                st.session_state["sell_stres_share_show_tables"] = show_tables
                st.session_state["sell_stres_share_table_results"] = results if show_tables else {}

                download_payload = {}
                if results:
                    combined_delta_df = pd.concat(
                        [df_delta.assign(ISIN=isin) for isin, df_delta in results.items()],
                        ignore_index=True,
                    )[["ISIN", "Q", "DeltaP"]]
                    download_payload["delta_csv"] = combined_delta_df.to_csv(index=False).encode("utf-8-sig")
                    download_payload["delta_xlsx"] = ss.dataframe_to_excel_bytes(
                        combined_delta_df, sheet_name="delta_p"
                    )
                    st.download_button(
                        label="💾 Скачать общий ΔP Excel",
                        data=ss.dataframe_to_excel_bytes(combined_delta_df, sheet_name="delta_p"),
                        file_name="sell_stres_share_deltaP_all.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                if meta_rows:
                    meta_df = pd.DataFrame(meta_rows, columns=["ISIN", "T", "Sigma", "MDTV"])
                    download_payload["meta_csv"] = meta_df.to_csv(index=False).encode("utf-8-sig")
                    download_payload["meta_xlsx"] = ss.dataframe_to_excel_bytes(meta_df, sheet_name="meta")
                    if show_tables:
                        st.session_state["sell_stres_share_meta_table"] = meta_df
                    else:
                        st.session_state["sell_stres_share_meta_table"] = None
                else:
                    st.session_state["sell_stres_share_meta_table"] = None

                st.session_state["sell_stres_share_downloads"] = download_payload if download_payload else None

        if st.session_state.get("sell_stres_share_show_tables") and st.session_state.get(
            "sell_stres_share_table_results"
        ):
            st.markdown("#### Результаты ΔP")
            for isin, df_delta in st.session_state["sell_stres_share_table_results"].items():
                st.markdown(f"**{isin}**")
                st.dataframe(df_delta, use_container_width=True)

        share_meta_table = st.session_state.get("sell_stres_share_meta_table")
        if share_meta_table is not None:
            st.markdown("#### Meta_mod")
            st.dataframe(share_meta_table, use_container_width=True)

        share_downloads = st.session_state.get("sell_stres_share_downloads")
        if share_downloads:
            if "delta_csv" in share_downloads:
                st.download_button(
                    label="💾 Скачать общий ΔP CSV",
                    data=share_downloads["delta_csv"],
                    file_name="sell_stres_share_deltaP_all.csv",
                    mime="text/csv",
                    key="share_delta_csv_dl",
                )
                st.download_button(
                    label="💾 Скачать общий ΔP Excel",
                    data=share_downloads["delta_xlsx"],
                    file_name="sell_stres_share_deltaP_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="share_delta_xlsx_dl",
                )
            if "meta_csv" in share_downloads:
                st.download_button(
                    label="💾 Скачать общий Meta_mod CSV",
                    data=share_downloads["meta_csv"],
                    file_name="sell_stres_share_meta_all.csv",
                    mime="text/csv",
                    key="share_meta_csv_dl",
                )
                st.download_button(
                    label="💾 Скачать общий Meta_mod Excel",
                    data=share_downloads["meta_xlsx"],
                    file_name="sell_stres_share_meta_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="share_meta_xlsx_dl",
                )
            render_email_compose_section("Sell_stres Share отчёт", "share_report", "sell_stres_share_deltaP_all.xlsx", share_downloads.get("delta_xlsx") if share_downloads else None)

    with bond_tab:
        st.markdown("### Bond")
        use_q_from_list_bond = st.checkbox(
            "Вводить Q для каждого ISIN (формат: ISIN | Q)", value=False, key="bond_q_per_isin"
        )
        if use_q_from_list_bond:
            bond_isin_q_input = st.text_area(
                "Введите ISIN и Q (каждая строка: ISIN | Q)",
                height=160,
                placeholder="RU000A1095L7 | 300000000\nRU000A0JX0J2 | 150000000",
                key="bond_isin_q_input",
            )
        else:
            bond_isin_input = st.text_area(
                "Введите или вставьте ISIN (через Ctrl+V, пробел или запятую)",
                height=160,
                placeholder="RU000A1095L7\nRU000A0JX0J2",
                key="bond_isin_input",
            )

        bond_c_value = st.number_input(
            "C (коэффициент влияния, 0–1)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="bond_c_value",
        )
        bond_date_from = st.date_input(
            "Дата начала (data_from)",
            value=datetime(2024, 1, 1).date(),
            key="bond_date_from",
        )
        bond_q_max = st.number_input(
            "Q_MAX (макс. объём продажи)",
            min_value=1,
            value=300_000_000,
            step=1_000_000,
            format="%d",
            key="bond_q_max",
            disabled=use_q_from_list_bond,
        )
        bond_use_log = st.checkbox(
            "Логарифмическое приближение", value=False, key="bond_q_log"
        )
        bond_q_mode = "log" if bond_use_log else "linear"

        if st.button("Рассчитать Sell_stres (Bond)", key="bond_calculate"):
            entries = []
            invalid_isins = []
            if use_q_from_list_bond:
                raw_lines = [line.strip() for line in bond_isin_q_input.splitlines() if line.strip()]
                for line in raw_lines:
                    parts = [p.strip() for p in re.split(r"[|;\t,]+", line) if p.strip()]
                    if not parts:
                        continue
                    isin = parts[0].upper()
                    q_val = parse_number(parts[1]) if len(parts) > 1 else None
                    if not isin_format_valid(isin):
                        invalid_isins.append(isin)
                        continue
                    if q_val is None or q_val <= 0:
                        st.warning(f"Некорректный Q для {isin}: {parts[1] if len(parts) > 1 else ''}")
                        continue
                    entries.append({"ISIN": isin, "Q_MAX": int(q_val)})
            else:
                raw_text = bond_isin_input.strip()
                if raw_text:
                    isins = re.split(r"[\s,;]+", raw_text)
                    isins = [i.strip().upper() for i in isins if i.strip()]
                    for isin in isins:
                        if not isin_format_valid(isin):
                            invalid_isins.append(isin)
                            continue
                        entries.append({"ISIN": isin, "Q_MAX": int(bond_q_max)})

            if invalid_isins:
                st.warning(
                    "Некорректные по формату ISIN пропущены: "
                    f"{', '.join(invalid_isins[:10])}{'...' if len(invalid_isins) > 10 else ''}"
                )
            if not entries:
                st.error("Нет валидных ISIN для обработки.")
            elif len(entries) > ss.MAX_SECURITIES_PER_RUN:
                st.error(
                    f"Слишком много бумаг за один запуск: {len(entries)}. "
                    f"Лимит: {ss.MAX_SECURITIES_PER_RUN}."
                )
            else:
                meta_rows = []
                results = {}
                progress_bar = st.progress(0.0)
                with st.spinner("Рассчитываем Sell_stres (Bond)..."):
                    for idx, entry in enumerate(entries, start=1):
                        isin = entry["ISIN"]
                        try:
                            delta_df, meta = ss.calculate_bond_delta_p(
                                request_get=request_get,
                                secid=isin,
                                c_value=float(bond_c_value),
                                date_from=bond_date_from.strftime("%Y-%m-%d"),
                                date_to=(datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d"),
                                q_mode=bond_q_mode,
                                q_max=entry["Q_MAX"],
                            )
                            results[isin] = delta_df
                            meta_rows.append(meta)
                        except Exception as exc:
                            st.error(f"{isin}: {exc}")
                        progress_bar.progress(idx / len(entries))

                show_tables = len(entries) == 1 and not use_q_from_list_bond
                st.session_state["sell_stres_bond_show_tables"] = show_tables
                st.session_state["sell_stres_bond_table_results"] = results if show_tables else {}

                download_payload = {}
                if results:
                    combined_delta_df = pd.concat(
                        [df_delta.assign(ISIN=isin) for isin, df_delta in results.items()],
                        ignore_index=True,
                    )[["ISIN", "Q", "DeltaP_pct"]]
                    download_payload["delta_csv"] = combined_delta_df.to_csv(index=False).encode("utf-8-sig")
                    download_payload["delta_xlsx"] = ss.dataframe_to_excel_bytes(
                        combined_delta_df, sheet_name="delta_p"
                    )
                    st.download_button(
                        label="💾 Скачать общий ΔP Excel (Bond)",
                        data=ss.dataframe_to_excel_bytes(combined_delta_df, sheet_name="delta_p"),
                        file_name="sell_stres_bond_deltaP_all.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                if meta_rows:
                    meta_df = pd.DataFrame(
                        meta_rows,
                        columns=["ISIN", "T", "SigmaY", "MDTV", "Price", "YTM", "Dmod"],
                    )
                    download_payload["meta_csv"] = meta_df.to_csv(index=False).encode("utf-8-sig")
                    download_payload["meta_xlsx"] = ss.dataframe_to_excel_bytes(meta_df, sheet_name="meta")
                    if show_tables:
                        st.session_state["sell_stres_bond_meta_table"] = meta_df
                    else:
                        st.session_state["sell_stres_bond_meta_table"] = None
                else:
                    st.session_state["sell_stres_bond_meta_table"] = None

                st.session_state["sell_stres_bond_downloads"] = download_payload if download_payload else None

        if st.session_state.get("sell_stres_bond_show_tables") and st.session_state.get(
            "sell_stres_bond_table_results"
        ):
            st.markdown("#### Результаты ΔP (Bond)")
            for isin, df_delta in st.session_state["sell_stres_bond_table_results"].items():
                st.markdown(f"**{isin}**")
                st.dataframe(df_delta, use_container_width=True)

        bond_meta_table = st.session_state.get("sell_stres_bond_meta_table")
        if bond_meta_table is not None:
            st.markdown("#### Meta_mod (Bond)")
            st.dataframe(bond_meta_table, use_container_width=True)

        bond_downloads = st.session_state.get("sell_stres_bond_downloads")
        if bond_downloads:
            if "delta_csv" in bond_downloads:
                st.download_button(
                    label="💾 Скачать общий ΔP CSV (Bond)",
                    data=bond_downloads["delta_csv"],
                    file_name="sell_stres_bond_deltaP_all.csv",
                    mime="text/csv",
                    key="bond_delta_csv_dl",
                )
                st.download_button(
                    label="💾 Скачать общий ΔP Excel (Bond)",
                    data=bond_downloads["delta_xlsx"],
                    file_name="sell_stres_bond_deltaP_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="bond_delta_xlsx_dl",
                )
            if "meta_csv" in bond_downloads:
                st.download_button(
                    label="💾 Скачать общий Meta_mod CSV (Bond)",
                    data=bond_downloads["meta_csv"],
                    file_name="sell_stres_bond_meta_all.csv",
                    mime="text/csv",
                    key="bond_meta_csv_dl",
                )
                st.download_button(
                    label="💾 Скачать общий Meta_mod Excel (Bond)",
                    data=bond_downloads["meta_xlsx"],
                    file_name="sell_stres_bond_meta_all.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="bond_meta_xlsx_dl",
                )
            render_email_compose_section("Sell_stres Bond отчёт", "bond_report", "sell_stres_bond_deltaP_all.xlsx", bond_downloads.get("delta_xlsx") if bond_downloads else None)

    st.stop()

# ---------------------------
# Index analytics view
# ---------------------------
if st.session_state["active_view"] == "index_analytics":
    open_index_analytics_sheet()
    st.stop()

# ---------------------------
# MOEX turnover view
# ---------------------------
if st.session_state["active_view"] == "moex_turnover":
    st.subheader("📊 MOEX turnover (trades)")
    st.markdown("Расчет оборота акции через MOEX ISS endpoint `history`.")

    secid_input = st.text_area(
        "SECID / ISIN / TICKER (каждый инструмент с новой строки)",
        value="",
        placeholder="SBER\nRU0009029540\nGAZP",
        key="moex_identifiers",
    )
    use_interval = st.checkbox("Считать за интервал дат", value=False, key="moex_use_interval")
    if use_interval:
        date_col_left, date_col_right = st.columns(2)
        with date_col_left:
            start_date_value = st.date_input(
                "START_DATE",
                value=datetime.now().date() - timedelta(days=30),
                key="moex_start_date",
            )
        with date_col_right:
            end_date_value = st.date_input(
                "END_DATE",
                value=datetime.now().date(),
                key="moex_end_date",
            )
        if end_date_value < start_date_value:
            st.error("END_DATE не может быть раньше START_DATE.")
            st.stop()
    else:
        single_date_value = st.date_input(
            "DATE",
            value=datetime.now().date() - timedelta(days=1),
            key="moex_single_date",
        )
        start_date_value = single_date_value
        end_date_value = single_date_value

    start_date_input = start_date_value.strftime("%Y-%m-%d")
    end_date_input = end_date_value.strftime("%Y-%m-%d")

    if st.button("Рассчитать оборот", key="calc_moex_turnover"):
        raw_identifiers = [line.strip().upper() for line in secid_input.splitlines() if line.strip()]
        if not raw_identifiers:
            st.error("Укажите хотя бы один SECID / ISIN / TICKER.")
        else:
            unique_identifiers = list(dict.fromkeys(raw_identifiers))
            with st.spinner("Загружаем сделки и считаем оборот..."):
                client = MoexTurnoverClient()
                report_rows = []
                errors = []
                board_rows = []

                for identifier in unique_identifiers:
                    try:
                        resolved_secid = resolve_identifier_to_secid(identifier)
                        if not resolved_secid:
                            raise ValueError(f"Инструмент '{identifier}' не найден на MOEX")
                        turnover = client.get_turnover(resolved_secid, start_date_input, end_date_input)
                    except Exception as exc:
                        errors.append(f"{identifier}: {exc}")
                        continue

                    report_rows.append(
                        {
                            "input": identifier,
                            "SECID": resolved_secid,
                            "TOTAL_regular": float(turnover["TOTAL_regular"]),
                            "TOTAL_SPEQ": float(turnover["TOTAL_SPEQ"]),
                            "TOTAL_NDM": float(turnover["TOTAL_NDM"]),
                            "TOTAL_all": float(turnover["TOTAL_all"]),
                        }
                    )
                    for board, value in turnover.get("board_turnover", {}).items():
                        board_rows.append(
                            {
                                "input": identifier,
                                "SECID": resolved_secid,
                                "board": board,
                                "turnover": float(value),
                            }
                        )

            if report_rows:
                st.success("Расчет завершен")
                report_df = pd.DataFrame(report_rows)
                st.dataframe(report_df, use_container_width=True)

                totals = report_df[["TOTAL_regular", "TOTAL_SPEQ", "TOTAL_NDM", "TOTAL_all"]].sum()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("TOTAL regular", f"{totals['TOTAL_regular']:,.2f}")
                col2.metric("TOTAL SPEQ", f"{totals['TOTAL_SPEQ']:,.2f}")
                col3.metric("TOTAL NDM", f"{totals['TOTAL_NDM']:,.2f}")
                col4.metric("TOTAL all", f"{totals['TOTAL_all']:,.2f}")

                if board_rows:
                    st.markdown("**Оборот по boards**")
                    board_df = pd.DataFrame(board_rows).sort_values(
                        ["input", "turnover"], ascending=[True, False]
                    )
                    st.dataframe(board_df, use_container_width=True)
                else:
                    st.info("Нет данных по boards за указанный период.")

            if errors:
                st.warning("Не удалось посчитать часть инструментов:")
                for error in errors:
                    st.write(f"- {error}")

            if not report_rows:
                st.error("Не удалось выполнить расчет ни для одного инструмента.")

    st.stop()


# ---------------------------
# Market statistics view
# ---------------------------
if st.session_state["active_view"] == "market_statistics":
    st.subheader("📈 Статистика рынка")
    st.markdown(
        "Расчет исторического объема торгов по списку инструментов с визуализацией и выгрузкой CSV/Excel."
    )

    market_mode = st.radio(
        "Тип инструментов",
        options=["Акции", "Облигации"],
        horizontal=True,
        key="market_statistics_mode",
    )
    market_kind = "shares" if market_mode == "Акции" else "bonds"

    identifiers_input = st.text_area(
        "SECID / ISIN / TICKER (каждый инструмент с новой строки)",
        value="",
        placeholder="SBER\nGAZP\nRU000A105SN8",
        key="market_statistics_identifiers",
    )
    use_all_papers = st.checkbox(
        "Считать статистику по всем бумагам выбранного рынка (без списка ISIN/SECID)",
        value=False,
        key="market_statistics_all_papers",
    )

    emitent_cache_key = f"market_statistics_emitent_map_{market_kind}"
    emitent_loaded_key = f"market_statistics_emitent_loaded_{market_kind}"
    if emitent_cache_key not in st.session_state:
        st.session_state[emitent_cache_key] = pd.DataFrame(columns=["SECID", "EMITENT_TITLE"])
    if emitent_loaded_key not in st.session_state:
        st.session_state[emitent_loaded_key] = False

    emitent_controls_col1, emitent_controls_col2 = st.columns([1, 2])
    with emitent_controls_col1:
        if st.button("📥 Загрузить список эмитентов", key=f"market_statistics_load_emitents_{market_kind}"):
            with st.spinner("Загружаем справочник эмитентов..."):
                st.session_state[emitent_cache_key] = load_security_emitents_map(market_kind)
                st.session_state[emitent_loaded_key] = True
    with emitent_controls_col2:
        st.caption("Список эмитентов загружается только по кнопке, чтобы не делать лишний запрос.")

    emitent_map_for_filter = st.session_state[emitent_cache_key]
    emitent_options = []
    if st.session_state[emitent_loaded_key] and not emitent_map_for_filter.empty:
        emitent_options = sorted(
            {
                normalize_emitent_title(title)
                for title in emitent_map_for_filter["EMITENT_TITLE"].dropna().astype(str).tolist()
            }
        )

    selected_emitents = st.multiselect(
        "Фильтр по эмитентам (если пусто — считаем по всем)",
        options=emitent_options,
        default=[],
        key="market_statistics_emitent_filter",
        disabled=not st.session_state[emitent_loaded_key],
    )
    if not st.session_state[emitent_loaded_key]:
        selected_emitents = []

    if st.session_state[emitent_loaded_key] and emitent_options:
        emitent_list_df = pd.DataFrame({"EMITENT_TITLE": emitent_options})
        st.download_button(
            label="💾 Выгрузить список эмитентов (CSV)",
            data=emitent_list_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"market_statistics_{market_kind}_emitent_list.csv",
            mime="text/csv",
            key=f"market_statistics_emitent_list_csv_{market_kind}",
        )
    elif st.session_state[emitent_loaded_key]:
        st.info("Список эмитентов для выбранного рынка пока недоступен.")

    date_col_left, date_col_right = st.columns(2)
    with date_col_left:
        market_start_date = st.date_input(
            "START_DATE",
            value=datetime.now().date() - timedelta(days=90),
            key="market_statistics_start_date",
        )
    with date_col_right:
        market_end_date = st.date_input(
            "END_DATE",
            value=datetime.now().date(),
            key="market_statistics_end_date",
        )

    if market_end_date < market_start_date:
        st.error("END_DATE не может быть раньше START_DATE.")
        st.stop()

    if st.button("Рассчитать статистику", key="calc_market_statistics"):
        raw_identifiers = [line.strip().upper() for line in identifiers_input.splitlines() if line.strip()]
        if not use_all_papers and not raw_identifiers:
            st.error("Укажите хотя бы один SECID / ISIN / TICKER.")
        else:
            full_rows = []
            errors = []
            with st.spinner("Загружаем историю торгов..."):
                if use_all_papers:
                    try:
                        hist_df = load_market_wide_history_values(
                            market_kind,
                            market_start_date.strftime("%Y-%m-%d"),
                            market_end_date.strftime("%Y-%m-%d"),
                        )
                        if hist_df.empty:
                            raise ValueError("Нет данных истории за указанный период")
                        hist_df["INPUT"] = "ALL_SECURITIES"
                        hist_df["ISIN"] = ""
                        emitent_map_df = st.session_state.get(
                            emitent_cache_key, pd.DataFrame(columns=["SECID", "EMITENT_TITLE"])
                        )
                        if st.session_state.get(emitent_loaded_key) and not emitent_map_df.empty:
                            hist_df = hist_df.merge(emitent_map_df, on="SECID", how="left")
                        if "EMITENT_TITLE" not in hist_df.columns:
                            hist_df["EMITENT_TITLE"] = ""
                        hist_df["EMITENT_TITLE"] = hist_df["EMITENT_TITLE"].map(normalize_emitent_title)
                        if selected_emitents:
                            hist_df = hist_df[hist_df["EMITENT_TITLE"].isin(selected_emitents)]
                        if hist_df.empty:
                            raise ValueError("Нет данных истории за указанный период/фильтр эмитентов")
                        full_rows.append(hist_df)
                    except Exception as exc:
                        errors.append(f"ALL_SECURITIES: {exc}")
                else:
                    unique_identifiers = list(dict.fromkeys(raw_identifiers))
                    for identifier in unique_identifiers:
                        try:
                            profile = resolve_market_security_profile(identifier, market_kind)
                            hist_df = load_market_history_values(
                                profile["secid"],
                                market_kind,
                                market_start_date.strftime("%Y-%m-%d"),
                                market_end_date.strftime("%Y-%m-%d"),
                            )
                            if hist_df.empty:
                                raise ValueError("Нет данных истории за указанный период")

                            hist_df["INPUT"] = profile["input"]
                            hist_df["ISIN"] = profile["isin"]
                            hist_df["EMITENT_TITLE"] = normalize_emitent_title(profile["emitent_title"])
                            if selected_emitents and hist_df["EMITENT_TITLE"].iloc[0] not in selected_emitents:
                                continue
                            full_rows.append(hist_df)
                        except Exception as exc:
                            errors.append(f"{identifier}: {exc}")

            if full_rows:
                combined_df = pd.concat(full_rows, ignore_index=True)
                combined_df = combined_df.sort_values(["TRADEDATE", "SECID"])

                st.success("Статистика рассчитана")

                totals = (
                    combined_df.groupby(["SECID", "ISIN", "SHORTNAME", "EMITENT_TITLE"], dropna=False, as_index=False)
                    .agg(
                        TOTAL_VALUE=("VALUE", "sum"),
                        TOTAL_NUMTRADES=("NUMTRADES", "sum"),
                        TOTAL_VOLUME=("VOLUME", "sum"),
                        DAYS=("TRADEDATE", "nunique"),
                    )
                    .sort_values("TOTAL_VALUE", ascending=False)
                )
                st.markdown("**Сводка по инструментам**")
                st.dataframe(totals, use_container_width=True)

                st.markdown("**Динамика общего оборота VALUE по дням**")
                daily_value_df = (
                    combined_df.groupby("TRADEDATE", as_index=False)["VALUE"].sum().sort_values("TRADEDATE")
                )
                st.line_chart(daily_value_df.set_index("TRADEDATE")["VALUE"], use_container_width=True)

                st.markdown("**Статистика по инструментам (график)**")
                chart_metric = st.selectbox(
                    "Показатель",
                    options=["TOTAL_VALUE", "TOTAL_NUMTRADES", "TOTAL_VOLUME"],
                    key="market_statistics_metric",
                )
                chart_data = totals[["SECID", chart_metric]].set_index("SECID")
                st.bar_chart(chart_data, use_container_width=True)

                st.markdown("**Агрегация по эмитентам**")
                emitent_df = (
                    combined_df.groupby("EMITENT_TITLE", dropna=False, as_index=False)
                    .agg(
                        TOTAL_VALUE=("VALUE", "sum"),
                        TOTAL_NUMTRADES=("NUMTRADES", "sum"),
                        TOTAL_VOLUME=("VOLUME", "sum"),
                    )
                    .sort_values("TOTAL_VALUE", ascending=False)
                )
                emitent_df["EMITENT_TITLE"] = emitent_df["EMITENT_TITLE"].map(normalize_emitent_title)
                st.dataframe(emitent_df, use_container_width=True)
                st.bar_chart(emitent_df.set_index("EMITENT_TITLE")[["TOTAL_VALUE"]], use_container_width=True)

                display_df = combined_df.copy()
                display_df["TRADEDATE"] = display_df["TRADEDATE"].dt.strftime("%Y-%m-%d")
                st.markdown("**Детальные данные (история)**")
                st.dataframe(display_df, use_container_width=True)

                daily_export_df = daily_value_df.copy()
                daily_export_df["TRADEDATE"] = daily_export_df["TRADEDATE"].dt.strftime("%Y-%m-%d")

                csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
                totals_csv_bytes = totals.to_csv(index=False).encode("utf-8-sig")
                daily_csv_bytes = daily_export_df.to_csv(index=False).encode("utf-8-sig")
                emitent_csv_bytes = emitent_df.to_csv(index=False).encode("utf-8-sig") if not emitent_df.empty else b""

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    display_df.to_excel(writer, index=False, sheet_name="history")
                    totals.to_excel(writer, index=False, sheet_name="totals")
                    daily_export_df.to_excel(writer, index=False, sheet_name="daily_value")
                    if not emitent_df.empty:
                        emitent_df.to_excel(writer, index=False, sheet_name="emitents")
                excel_buffer.seek(0)

                st.markdown("### Скачать отчеты")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="💾 История (CSV)",
                        data=csv_bytes,
                        file_name=f"market_statistics_{market_kind}_history.csv",
                        mime="text/csv",
                        key="market_statistics_history_csv",
                    )
                    st.download_button(
                        label="💾 Сводка (CSV)",
                        data=totals_csv_bytes,
                        file_name=f"market_statistics_{market_kind}_totals.csv",
                        mime="text/csv",
                        key="market_statistics_totals_csv",
                    )
                with dl_col2:
                    st.download_button(
                        label="💾 Динамика по дням (CSV)",
                        data=daily_csv_bytes,
                        file_name=f"market_statistics_{market_kind}_daily.csv",
                        mime="text/csv",
                        key="market_statistics_daily_csv",
                    )
                    st.download_button(
                        label="💾 Полный отчёт (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name=f"market_statistics_{market_kind}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="market_statistics_excel",
                    )
                if emitent_csv_bytes:
                    st.download_button(
                        label="💾 По эмитентам (CSV)",
                        data=emitent_csv_bytes,
                        file_name=f"market_statistics_{market_kind}_emitents.csv",
                        mime="text/csv",
                        key="market_statistics_emitent_csv",
                    )

            if errors:
                st.warning("Не удалось обработать часть инструментов:")
                for error in errors:
                    st.write(f"- {error}")

            if not full_rows:
                st.error("Не удалось получить статистику ни для одного инструмента.")

    st.stop()


# ---------------------------
# Turnover export view
# ---------------------------
if st.session_state["active_view"] == "turnover_export":
    st.subheader("📊 Выгрузка оборотов по акциям и облигациям")
    st.markdown("Выгрузка оборотов за период по списку бумаг в Excel с опциональной статистикой ликвидности.")

    turnover_mode = st.radio(
        "Тип инструментов",
        options=["Акции", "Облигации"],
        horizontal=True,
        key="turnover_export_mode",
    )
    market_kind = "shares" if turnover_mode == "Акции" else "bonds"

    identifiers_input = st.text_area(
        "ISIN / TICKER / SECID (каждый инструмент с новой строки)",
        value="",
        placeholder="SBER\nGAZP\nRU000A105SN8",
        key="turnover_export_identifiers",
    )

    include_otc = st.checkbox(
        "Выгрузка внебиржевых оборотов (NDM)",
        value=False,
        key="turnover_export_include_otc",
        help="По аналогии с MOEX turnover: при включении добавляется NDM.",
    )

    show_stats_settings = st.checkbox(
        "Показать настройки статистики ликвидности",
        value=False,
        key="turnover_export_show_stats",
    )
    stat_adtv = False
    stat_mdtv = False
    stat_sigma = False
    if show_stats_settings:
        st.caption("Выберите только нужные метрики, чтобы не перегружать отчёт.")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            stat_adtv = st.checkbox("ADTV", value=True, key="turnover_export_stat_adtv")
        with stat_col2:
            stat_mdtv = st.checkbox("MDTV", value=True, key="turnover_export_stat_mdtv")
        with stat_col3:
            stat_sigma = st.checkbox("SIGMA", value=True, key="turnover_export_stat_sigma")

    calculate_stats = stat_adtv or stat_mdtv or stat_sigma

    date_col_left, date_col_right = st.columns(2)
    with date_col_left:
        start_date = st.date_input(
            "START_DATE",
            value=datetime.now().date() - timedelta(days=90),
            key="turnover_export_start_date",
        )
    with date_col_right:
        end_date = st.date_input(
            "END_DATE",
            value=datetime.now().date(),
            key="turnover_export_end_date",
        )

    if end_date < start_date:
        st.error("END_DATE не может быть раньше START_DATE.")
        st.stop()

    if st.button("Сформировать выгрузку", key="turnover_export_run"):
        raw_identifiers = [line.strip().upper() for line in identifiers_input.splitlines() if line.strip()]
        if not raw_identifiers:
            st.error("Укажите хотя бы один ISIN / TICKER / SECID.")
        else:
            report_rows = []
            daily_rows = []
            errors = []

            with st.spinner("Загружаем обороты..."):
                for identifier in list(dict.fromkeys(raw_identifiers)):
                    try:
                        profile = resolve_market_security_profile(identifier, market_kind)
                        daily_df, totals = load_turnover_components_via_iss(
                            profile["secid"],
                            market_kind,
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )

                        daily_df["INPUT"] = profile["input"]
                        daily_df["ISIN"] = profile["isin"]
                        daily_df["SHORTNAME"] = profile["shortname"]
                        daily_rows.append(daily_df)

                        total_regular = float(totals["TOTAL_regular"])
                        total_speq = float(totals["TOTAL_SPEQ"])
                        total_ndm = float(totals["TOTAL_NDM"])
                        total_exchange = total_regular + total_speq
                        total_selected = total_exchange + total_ndm if include_otc else total_exchange

                        report_rows.append(
                            {
                                "INPUT": profile["input"],
                                "SECID": profile["secid"],
                                "ISIN": profile["isin"],
                                "SHORTNAME": profile["shortname"],
                                "TOTAL_REGULAR": total_regular,
                                "TOTAL_SPEQ": total_speq,
                                "TOTAL_NDM": total_ndm,
                                "TOTAL_EXCHANGE": total_exchange,
                                "TOTAL_SELECTED": total_selected,
                            }
                        )
                    except Exception as exc:
                        errors.append(f"{identifier}: {exc}")

            if report_rows:
                report_df = pd.DataFrame(report_rows).sort_values("TOTAL_SELECTED", ascending=False)
                st.success("Выгрузка сформирована")
                st.dataframe(report_df, use_container_width=True)

                detailed_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
                stats_df = pd.DataFrame()
                if not detailed_df.empty:
                    detailed_df["TRADEDATE"] = pd.to_datetime(detailed_df["TRADEDATE"], errors="coerce")
                    detailed_df = detailed_df.dropna(subset=["TRADEDATE"])
                    detailed_df["VALUE"] = (
                        pd.to_numeric(detailed_df["REGULAR"], errors="coerce").fillna(0.0)
                        + pd.to_numeric(detailed_df["SPEQ"], errors="coerce").fillna(0.0)
                    )
                    if include_otc:
                        detailed_df["VALUE"] += pd.to_numeric(detailed_df["NDM"], errors="coerce").fillna(0.0)

                    if calculate_stats:
                        stats_df = calculate_turnover_liquidity_stats(detailed_df)
                        stats_df = stats_df.merge(
                            report_df[["SECID", "INPUT", "ISIN", "SHORTNAME"]].drop_duplicates(),
                            on="SECID",
                            how="left",
                        )
                        selected_stats_cols = []
                        if stat_adtv:
                            selected_stats_cols.append("ADTV")
                        if stat_mdtv:
                            selected_stats_cols.append("MDTV")
                        if stat_sigma:
                            selected_stats_cols.append("SIGMA")

                        stats_df = stats_df[["INPUT", "SECID", "ISIN", "SHORTNAME", *selected_stats_cols]]
                        st.markdown("**Статистика ликвидности**")
                        st.dataframe(stats_df, use_container_width=True)

                    chart_df = detailed_df.groupby("TRADEDATE", as_index=False)["VALUE"].sum().sort_values("TRADEDATE")
                    st.markdown("**Динамика оборотов по дням**")
                    st.line_chart(chart_df.set_index("TRADEDATE")["VALUE"], use_container_width=True)

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    report_df.to_excel(writer, index=False, sheet_name="turnover_totals")

                    if detailed_df.empty:
                        pd.DataFrame(columns=["TRADEDATE", "SECID", "VALUE"]).to_excel(
                            writer, index=False, sheet_name="turnover_daily"
                        )
                    else:
                        export_df = detailed_df.copy()
                        export_df["TRADEDATE"] = export_df["TRADEDATE"].dt.strftime("%Y-%m-%d")
                        export_df.to_excel(writer, index=False, sheet_name="turnover_daily")

                    if calculate_stats and not stats_df.empty:
                        stats_df.to_excel(writer, index=False, sheet_name="liquidity_stats")
                excel_buffer.seek(0)

                st.download_button(
                    label="💾 Скачать Excel с оборотами",
                    data=excel_buffer.getvalue(),
                    file_name=f"turnover_export_{market_kind}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="turnover_export_excel",
                )

                if calculate_stats and not stats_df.empty:
                    stats_excel_buffer = BytesIO()
                    with pd.ExcelWriter(stats_excel_buffer, engine="openpyxl") as stats_writer:
                        stats_df.to_excel(stats_writer, index=False, sheet_name="liquidity_stats")
                    stats_excel_buffer.seek(0)
                    st.download_button(
                        label="💾 Скачать статистические показатели (Excel)",
                        data=stats_excel_buffer.getvalue(),
                        file_name=f"turnover_stats_{market_kind}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="turnover_export_stats_excel",
                    )

            if errors:
                st.warning("Не удалось обработать часть инструментов:")
                for error in errors:
                    st.write(f"- {error}")

            if not report_rows:
                st.error("Не удалось получить обороты ни для одного инструмента.")

    st.stop()


# ---------------------------
# REPO duration settings
# ---------------------------
st.subheader("⚙️ Настройки длительности РЕПО")
if "overnight" not in st.session_state:
    st.session_state["overnight"] = False
if "extra_days" not in st.session_state:
    st.session_state["extra_days"] = 2

if st.button("🔄 Очистить форму"):
    st.session_state["overnight"] = False
    st.session_state["extra_days"] = 2
    st.session_state["results"] = None
    st.session_state["file_loaded"] = False
    st.session_state["last_file_name"] = None
    trigger_rerun()

overnight = st.checkbox("Overnight РЕПО", key="overnight")
extra_days_input = st.number_input(
    "Дней РЕПО:",
    min_value=2,
    max_value=366,
    step=1,
    disabled=st.session_state["overnight"],
    key="extra_days",
)
if st.session_state["overnight"]:
    st.markdown(
        "<span style='color:gray'>Дополнительные дни отключены при включенном Overnight</span>",
        unsafe_allow_html=True,
    )
days_threshold = 2 if st.session_state["overnight"] else 1 + st.session_state["extra_days"]
st.write(f"Текущее значение границы выплат: {days_threshold} дн.")

# ---------------------------
# UI: input tabs
# ---------------------------
st.subheader("📤 Загрузка или ввод ISIN")
tab1, tab2 = st.tabs(["📁 Загрузить файл", "✍️ Ввести вручную"])

with tab1:
    uploaded_file = st.file_uploader("Загрузите Excel или CSV с колонкой ISIN", type=["xlsx", "xls", "csv"])
    st.write("Пример шаблона (скачайте и заполните колонку ISIN):")
    sample_csv = "ISIN\nRU000A0JX0J2\nRU000A0ZZZY1\n"
    st.download_button("Скачать шаблон CSV", data=sample_csv, file_name="template_isin.csv", mime="text/csv")

with tab2:
    isin_input = st.text_area("Введите или вставьте ISIN (через Ctrl+V, пробел или запятую)", height=150)
    if st.button("🔍 Получить данные по введённым ISIN"):
        raw_text = isin_input.strip()
        if raw_text:
            isins = re.split(r"[\s,;]+", raw_text)
            isins = [i.strip().upper() for i in isins if i.strip()]
            invalid_format = [i for i in isins if not isin_format_valid(i)]
            invalid_checksum = [i for i in isins if isin_format_valid(i) and not isin_checksum_valid(i)]
            if invalid_format:
                st.warning(
                    f"Некорректные по формату ISIN будут пропущены: {', '.join(invalid_format[:10])}"
                    f"{'...' if len(invalid_format) > 10 else ''}"
                )
            if invalid_checksum:
                st.info(
                    f"ISIN с неверной контрольной суммой (будут пропущены): {', '.join(invalid_checksum[:10])}"
                    f"{'...' if len(invalid_checksum) > 10 else ''}"
                )
            isins = [i for i in isins if isin_format_valid(i) and isin_checksum_valid(i)]
            if not isins:
                st.error("Нет валидных ISIN для обработки.")
            else:
                with st.spinner("Запрос данных..."):
                    results = fetch_isins(isins, show_progress=True)
                st.session_state["results"] = pd.DataFrame(results)
                st.success("✅ Данные успешно получены!")

# ---------------------------
# File upload handling
# ---------------------------
if uploaded_file:
    if not st.session_state["file_loaded"] or uploaded_file.name != st.session_state["last_file_name"]:
        st.session_state["file_loaded"] = True
        st.session_state["last_file_name"] = uploaded_file.name

        df = safe_read_filelike(uploaded_file)
        if df.empty:
            st.error("❌ Не удалось прочитать файл или файл пуст.")
            st.stop()
        df.columns = [c.strip().upper() for c in df.columns]

        if "ISIN" not in df.columns:
            candidates = []
            for c in df.columns:
                try:
                    if df[c].dropna().astype(str).str.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$").any():
                        candidates.append(c)
                except Exception:
                    continue
            if len(candidates) == 1:
                df.rename(columns={candidates[0]: "ISIN"}, inplace=True)
                st.info(f"Авто-детект: колонка '{candidates[0]}' использована как ISIN")
            else:
                st.error("❌ В файле должна быть колонка 'ISIN' или одна колонка с ISIN-подобными значениями.")
                st.stop()

        isins = df["ISIN"].dropna().unique().tolist()
        isins = [str(x).strip().upper() for x in isins if str(x).strip()]
        invalid_fmt = [i for i in isins if not isin_format_valid(i)]
        invalid_chk = [i for i in isins if isin_format_valid(i) and not isin_checksum_valid(i)]
        if invalid_fmt:
            st.warning(
                f"Некорректные по формату ISIN пропущены: {', '.join(invalid_fmt[:10])}"
                f"{'...' if len(invalid_fmt) > 10 else ''}"
            )
        if invalid_chk:
            st.info(
                f"ISIN с неверной контрольной суммой пропущены: {', '.join(invalid_chk[:10])}"
                f"{'...' if len(invalid_chk) > 10 else ''}"
            )
        isins = [i for i in isins if isin_format_valid(i) and isin_checksum_valid(i)]

        st.write(f"Найдено {len(isins)} валидных уникальных ISIN для обработки.")
        if isins:
            with st.spinner("Запрос данных по файлу..."):
                results = fetch_isins(isins, show_progress=True)
            st.session_state["results"] = pd.DataFrame(results)
            st.success("✅ Данные успешно получены из файла!")

# ---------------------------
# Load emitter reference (optional)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_emitter_names():
    url = "https://raw.githubusercontent.com/mainarkler/Bond_date/refs/heads/main/Pifagr_name_with_emitter.csv"
    try:
        df_emitters = pd.read_csv(url, dtype=str)
        df_emitters.columns = [c.strip() for c in df_emitters.columns]
        return df_emitters
    except Exception:
        return pd.DataFrame(columns=["Issuer", "EMITTER_ID"])


df_emitters = fetch_emitter_names()

# ---------------------------
# Styling helper
# ---------------------------
def style_df(row):
    if pd.isna(row.get("Наименование инструмента")) or row.get("Наименование инструмента") in [None, "", "None"]:
        return ["background-color: DimGray; color: white"] * len(row)
    today = datetime.today().date()
    danger_threshold = today + timedelta(days=days_threshold)
    key_dates = ["Дата погашения", "Дата оферты Put", "Дата оферты Call", "Дата фиксации купона", "Дата купона"]
    colors = ["" for _ in row]
    for i, col in enumerate(row.index):
        if col in key_dates and pd.notnull(row[col]):
            try:
                d = pd.to_datetime(row[col]).date()
                if d <= danger_threshold:
                    colors[i] = "background-color: Chocolate"
            except Exception:
                pass
    if any(c == "background-color: Chocolate" for c in colors):
        colors = ["background-color: SandyBrown" if c == "" else c for c in colors]
    return colors

# ---------------------------
# Show results (table + export) with filter for orange-highlighted rows
# ---------------------------
if st.session_state["results"] is not None:
    df_res = st.session_state["results"].copy()

    if "Код эмитента" in df_res.columns and not df_emitters.empty:
        try:
            df_res = df_res.merge(df_emitters, how="left", left_on="Код эмитента", right_on="EMITTER_ID")
            df_res["Эмитент"] = df_res.get("Issuer")
            df_res.drop(columns=["Issuer", "EMITTER_ID"], inplace=True, errors="ignore")
            cols = df_res.columns.tolist()
            if "Эмитент" in cols and "Код эмитента" in cols:
                cols.remove("Эмитент")
                idx = cols.index("Код эмитента")
                cols.insert(idx + 1, "Эмитент")
                df_res = df_res[cols]
            st.session_state["results"] = df_res
        except Exception:
            pass
    else:
        st.warning("⚠️ В данных нет колонки 'Код эмитента' — объединение со справочником пропущено.")

    st.markdown(f"**Всего записей:** {len(df_res)}")

    today = datetime.today().date()
    danger_threshold = today + timedelta(days=days_threshold)
    key_dates = ["Дата погашения", "Дата оферты Put", "Дата оферты Call", "Дата фиксации купона", "Дата купона"]

    mask_any = pd.Series(False, index=df_res.index)
    for col in key_dates:
        if col in df_res.columns:
            try:
                s = pd.to_datetime(df_res[col], errors="coerce").dt.date
                mask_any = mask_any | (s <= danger_threshold)
            except Exception:
                pass

    only_orange = st.checkbox("Показать бумаги с отсечкой в периоде", value=False)
    if only_orange:
        df_show = df_res[mask_any].copy()
        st.markdown(f"**Показано записей с отсечкой:** {len(df_show)}")
        if df_show.empty:
            st.info("Нет бумаг, попадающих под критерий (отсечки).")
    else:
        df_show = df_res

    st.dataframe(df_show.style.apply(style_df, axis=1), use_container_width=True)

    def to_excel_bytes(df: pd.DataFrame):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Данные")
        return output.getvalue()

    def to_csv_bytes(df: pd.DataFrame):
        return df.to_csv(index=False).encode("utf-8-sig")


    repo_xlsx = to_excel_bytes(df_show)
    st.download_button(
        label="💾 Скачать результат (Excel)",
        data=repo_xlsx,
        file_name="bond_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        label="💾 Скачать результат (CSV)",
        data=to_csv_bytes(df_show),
        file_name="bond_data.csv",
        mime="text/csv",
    )

    render_email_compose_section("Отчёт по облигациям", "repo_report", "bond_data.xlsx", repo_xlsx)
else:
    st.info("👆 Загрузите файл или введите ISIN-ы вручную.")
