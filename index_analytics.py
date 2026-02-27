from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


INDEX_ANALYTICS_HELP_TEXT = (
    "Загрузка данных из ISS MOEX `/statistics/engines/stock/markets/index/analytics/{INDEX}` "
    "с фильтром по дате и связкой тикер → ISIN."
)


@st.cache_data(ttl=1800)
def ticker_to_isin_cached(ticker: str, _request_get_func) -> Optional[str]:
    secid = str(ticker).strip().upper()
    if not secid:
        return None

    response = _request_get_func(
        "https://iss.moex.com/iss/securities.json",
        params={"q": secid, "iss.meta": "off"},
        timeout=30,
    )
    js = response.json()
    rows = js.get("securities", {}).get("data", [])
    cols = js.get("securities", {}).get("columns", [])
    if not rows or not cols:
        return None

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return None

    secid_col = "secid" if "secid" in df.columns else "SECID"
    isin_col = "isin" if "isin" in df.columns else "ISIN"
    if secid_col not in df.columns or isin_col not in df.columns:
        return None

    match = df[df[secid_col].astype(str).str.upper() == secid]
    if match.empty:
        return None

    isin = match[isin_col].iloc[0]
    return str(isin).strip().upper() if pd.notna(isin) and str(isin).strip() else None


def _empty_index_weights_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "ISIN", "Tiker", "Weight"])


def _to_df(block: dict) -> pd.DataFrame:
    if not isinstance(block, dict):
        return pd.DataFrame()

    rows = block.get("data", [])
    cols = block.get("columns", [])
    if not rows or not cols:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=cols)


def _normalize_snapshot_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Date", "Tiker", "Weight"])

    date_col = "tradedate" if "tradedate" in raw_df.columns else "TRADEDATE"
    weight_col = "weight" if "weight" in raw_df.columns else "WEIGHT"

    if date_col not in raw_df.columns or weight_col not in raw_df.columns:
        return pd.DataFrame(columns=["Date", "Tiker", "Weight"])

    ticker_col = None
    for col in ("ticker", "secid", "shortname", "TICKER", "SECID"):
        if col in raw_df.columns:
            ticker_col = col
            break

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(raw_df[date_col], errors="coerce")
    if ticker_col is None:
        out["Tiker"] = ""
    else:
        out["Tiker"] = raw_df[ticker_col].astype(str).str.strip().str.upper()

    out["Weight"] = pd.to_numeric(raw_df[weight_col], errors="coerce")
    return out.dropna(subset=["Date", "Weight"])


def _extract_prev_date(cursor_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if cursor_df.empty:
        return None

    for col in ("PREV_DATE", "prev_date", "PREVDATE", "prevdate"):
        if col in cursor_df.columns:
            prev_value = cursor_df[col].iloc[0]
            prev_dt = pd.to_datetime(prev_value, errors="coerce")
            if pd.notna(prev_dt):
                return prev_dt.normalize()
    return None


def _extract_total_and_pagesize(cursor_df: pd.DataFrame, fetched_count: int) -> Tuple[int, int]:
    if cursor_df.empty:
        return fetched_count, fetched_count

    def _get_int(columns: Tuple[str, ...], default: int) -> int:
        for col in columns:
            if col in cursor_df.columns:
                value = pd.to_numeric(cursor_df[col].iloc[0], errors="coerce")
                if pd.notna(value):
                    return int(value)
        return default

    total = _get_int(("TOTAL", "total"), fetched_count)
    pagesize = _get_int(("PAGESIZE", "pagesize"), max(fetched_count, 1))
    return max(total, fetched_count), max(pagesize, 1)


@st.cache_data(ttl=1800)
def _fetch_index_snapshot(index_name: str, date_str: str, _request_get_func) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/"
        f"{index_name}.json"
    )

    start = 0
    chunks: List[pd.DataFrame] = []
    prev_date = None
    total = None

    while True:
        response = _request_get_func(
            url,
            timeout=60,
            params={"iss.meta": "off", "limit": 100, "date": date_str, "start": start},
        )
        js = response.json()

        analytics_df = _to_df(js.get("analytics", {}))
        cursor_df = _to_df(js.get("analytics.cursor", {}))

        normalized = _normalize_snapshot_df(analytics_df)
        if not normalized.empty:
            chunks.append(normalized)

        if prev_date is None:
            prev_date = _extract_prev_date(cursor_df)

        fetched_now = len(analytics_df)
        if total is None:
            total, pagesize = _extract_total_and_pagesize(cursor_df, fetched_now)
        else:
            _, pagesize = _extract_total_and_pagesize(cursor_df, fetched_now)

        if fetched_now == 0:
            break

        start += fetched_now
        if start >= total or fetched_now < pagesize:
            break

    if not chunks:
        return pd.DataFrame(columns=["Date", "Tiker", "Weight"]), prev_date

    snapshot_df = pd.concat(chunks, ignore_index=True)
    snapshot_df = snapshot_df.drop_duplicates(subset=["Date", "Tiker"], keep="last")
    return snapshot_df, prev_date


def fetch_index_weights(
    request_get,
    index_name: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    index_name = str(index_name).strip().upper()
    if not index_name:
        raise ValueError("Укажите код индекса, например IMOEX")

    end_dt = pd.to_datetime(date_to if date_to else datetime.today().date(), errors="coerce")
    start_dt = pd.to_datetime(date_from if date_from else end_dt, errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("Некорректные даты периода")

    start_dt = start_dt.normalize()
    end_dt = end_dt.normalize()

    if start_dt > end_dt:
        raise ValueError("Дата 'с' не может быть больше даты 'по'.")

    single_date_mode = start_dt == end_dt

    collected: List[pd.DataFrame] = []
    seen_dates: Set[pd.Timestamp] = set()

    current_dt = end_dt
    while True:
        if current_dt < start_dt and not (single_date_mode and not collected):
            break
        snapshot_df, prev_dt = _fetch_index_snapshot(
            index_name=index_name,
            date_str=current_dt.strftime("%Y-%m-%d"),
            _request_get_func=request_get,
        )

        if snapshot_df.empty:
            if prev_dt is not None and prev_dt < current_dt:
                current_dt = prev_dt
                continue
            break

        effective_dt = snapshot_df["Date"].max().normalize()
        if effective_dt < start_dt and not (single_date_mode and not collected):
            break

        if effective_dt not in seen_dates:
            day_df = snapshot_df[snapshot_df["Date"].dt.normalize() == effective_dt]
            if not day_df.empty:
                collected.append(day_df)
                seen_dates.add(effective_dt)

        if prev_dt is None or prev_dt >= current_dt:
            break

        current_dt = prev_dt

    if not collected:
        return _empty_index_weights_df()

    df = pd.concat(collected, ignore_index=True)
    if single_date_mode:
        requested_day = start_dt.date()
        same_day_df = df[df["Date"].dt.date == requested_day]
        if same_day_df.empty:
            fallback_day = df["Date"].max().normalize()
            df = df[df["Date"].dt.normalize() == fallback_day]
        else:
            df = same_day_df
    else:
        df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]

    if df.empty:
        return _empty_index_weights_df()

    df = df.drop_duplicates(subset=["Date", "Tiker"], keep="last")
    df["ISIN"] = df["Tiker"].apply(lambda t: ticker_to_isin_cached(t, request_get))

    out = df[["Date", "ISIN", "Tiker", "Weight"]].copy()
    out["Date"] = out["Date"].dt.date
    out["ISIN"] = out["ISIN"].fillna("")
    return out.sort_values(["Date", "Tiker"]).reset_index(drop=True)


def render_index_analytics_view(request_get, dataframe_to_excel_bytes):
    if "index_matrix_df" not in st.session_state:
        st.session_state["index_matrix_df"] = None
    if "index_weight_matrix" not in st.session_state:
        st.session_state["index_weight_matrix"] = None
    if "index_last_code" not in st.session_state:
        st.session_state["index_last_code"] = "IMOEX"

    st.subheader("🧾 Весы индекса MOEX")
    st.markdown(INDEX_ANALYTICS_HELP_TEXT)

    idx_col1, idx_col2 = st.columns([1.4, 1])
    with idx_col1:
        index_code = st.text_input(
            "Код индекса",
            value="",
            placeholder="IMOEX",
            help="Например: IMOEX, RTSI",
            key="idx_code_input",
        )
    with idx_col2:
        load_period = st.checkbox(
            "Загружать за период",
            value=False,
            help="По умолчанию загружается состав на одну дату.",
            key="idx_use_period",
        )

    st.caption("Если код индекса не указан, используется IMOEX.")

    if load_period:
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            date_from = st.date_input(
                "Дата с",
                value=datetime.today().date() - timedelta(days=30),
                key="idx_date_from",
            )
        with date_col2:
            date_to = st.date_input(
                "Дата по",
                value=datetime.today().date(),
                key="idx_date_to",
            )
    else:
        single_date = st.date_input(
            "Дата",
            value=datetime.today().date(),
            key="idx_single_date",
        )
        date_from = single_date
        date_to = single_date

    if st.button("Загрузить данные индекса", key="load_index_analytics"):
        index_code = (index_code or "IMOEX").strip().upper()
        if date_from > date_to:
            st.error("Дата 'с' не может быть больше даты 'по'.")
        else:
            with st.spinner("Формируется..."):
                try:
                    df_index = fetch_index_weights(
                        request_get=request_get,
                        index_name=index_code,
                        date_from=date_from.strftime("%Y-%m-%d"),
                        date_to=date_to.strftime("%Y-%m-%d"),
                    )
                    st.session_state["index_matrix_df"] = df_index
                    st.session_state["index_weight_matrix"] = None
                    st.session_state["index_last_code"] = index_code
                    st.success(f"Загружено строк: {len(df_index)}")
                except Exception as exc:
                    st.error(f"Ошибка загрузки индекса: {exc}")

    current_df = st.session_state.get("index_matrix_df")
    if current_df is None:
        return

    last_index_code = st.session_state.get("index_last_code", "IMOEX").upper()

    st.markdown("#### Таблица состава")
    st.dataframe(current_df, use_container_width=True)

    st.download_button(
        label="💾 Скачать таблицу (CSV)",
        data=current_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"index_weights_{last_index_code}.csv",
        mime="text/csv",
        key="index_weights_csv_dl",
    )

    if st.button("Сформировать матрицу: строки — тикеры, столбцы — даты", key="build_index_weight_matrix"):
        matrix_df = current_df.pivot_table(
            index="Tiker",
            columns="Date",
            values="Weight",
            aggfunc="first",
        ).sort_index()
        matrix_df = matrix_df.reindex(sorted(matrix_df.columns), axis=1).reset_index()
        st.session_state["index_weight_matrix"] = matrix_df

    matrix_df = st.session_state.get("index_weight_matrix")
    if matrix_df is None:
        return

    st.markdown("#### Матрица весов")
    st.dataframe(matrix_df, use_container_width=True)
    st.download_button(
        label="💾 Скачать матрицу (Excel)",
        data=dataframe_to_excel_bytes(matrix_df, sheet_name="index_matrix"),
        file_name=f"index_weight_matrix_{last_index_code}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="index_matrix_xlsx_dl",
    )

