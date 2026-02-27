from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


@st.cache_data(ttl=1800)
def ticker_to_isin_cached(ticker: str, _request_get_func) -> str | None:
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


def _parse_index_rows(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = [node.attrib for node in root.findall(".//row")]

    start = 0
    chunks: list[pd.DataFrame] = []
    prev_date = None
    total = None

    while True:
        response = _request_get_func(
            url,
            timeout=60,
            params={"iss.meta": "off", "limit": 100, "date": date_str, "start": start},
        )
        js = response.json()

    df["Date"] = pd.to_datetime(df["tradedate"], errors="coerce")
    ticker_col = "ticker" if "ticker" in df.columns else "secids"
    if ticker_col not in df.columns:
        df["Tiker"] = ""
    else:
        df["Tiker"] = df[ticker_col].astype(str).str.strip().str.upper()

    df["Weight"] = pd.to_numeric(df.get("weight"), errors="coerce")
    return df.dropna(subset=["Date", "Weight"])


@st.cache_data(ttl=1800)
def _fetch_index_rows_for_date(index_name: str, date_str: str, _request_get_func) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/"
        f"{index_name}.xml"
    )

    for date_param_name in ("date", "tradedate"):
        response = _request_get_func(
            url,
            timeout=60,
            params={"iss.meta": "off", "limit": 10000, date_param_name: date_str},
        )
        parsed = _parse_index_rows(response.content)
        if not parsed.empty:
            return parsed

    return pd.DataFrame(columns=["Date", "Tiker", "Weight"])


def fetch_index_weights(request_get, index_name: str, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    index_name = str(index_name).strip().upper()
    if not index_name:
        raise ValueError("Укажите код индекса, например IMOEX")

    if date_from and date_to:
        start_dt = pd.to_datetime(date_from)
        end_dt = pd.to_datetime(date_to)
        dates = pd.date_range(start=start_dt, end=end_dt, freq="D")
        date_frames = []
        for single_date in dates:
            fetched = _fetch_index_rows_for_date(
                index_name=index_name,
                date_str=single_date.strftime("%Y-%m-%d"),
                _request_get_func=request_get,
            )
            if fetched.empty:
                continue
            fetched = fetched[fetched["Date"].dt.date == single_date.date()]
            if not fetched.empty:
                date_frames.append(fetched)

        if date_frames:
            df = pd.concat(date_frames, ignore_index=True)
        else:
            return pd.DataFrame(columns=["Date", "ISIN", "Tiker", "Weight"])
    else:
        url = (
            "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/"
            f"{index_name}.xml"
        )
        response = request_get(url, timeout=60, params={"iss.meta": "off", "limit": 10000})
        df = _parse_index_rows(response.content)

    df["ISIN"] = df["Tiker"].apply(lambda t: ticker_to_isin_cached(t, request_get))
    df = df.drop_duplicates(subset=["Date", "Tiker"], keep="last")

    out = df[["Date", "ISIN", "Tiker", "Weight"]].copy()
    out["Date"] = out["Date"].dt.date
    out["ISIN"] = out["ISIN"].fillna("")
    return out.sort_values(["Date", "Tiker"]).reset_index(drop=True)


def render_index_analytics_view(request_get, dataframe_to_excel_bytes):
    if "index_matrix_df" not in st.session_state:
        st.session_state["index_matrix_df"] = None
    if "index_weight_matrix" not in st.session_state:
        st.session_state["index_weight_matrix"] = None

    st.subheader("🧾 Весы индекса MOEX")
    st.markdown(
        "Загрузка данных из ISS MOEX `/statistics/engines/stock/markets/index/analytics/{INDEX}` "
        "с фильтром по дате и связкой тикер → ISIN."
    )

    idx_col1, idx_col2, idx_col3 = st.columns([1.2, 1, 1])
    with idx_col1:
        index_code = st.text_input("Код индекса", value="IMOEX", help="Например: IMOEX, RTSI")
    with idx_col2:
        date_from = st.date_input("Дата с", value=datetime.today().date() - timedelta(days=30), key="idx_date_from")
    with idx_col3:
        date_to = st.date_input("Дата по", value=datetime.today().date(), key="idx_date_to")

    if st.button("Загрузить данные индекса", key="load_index_analytics"):
        if date_from > date_to:
            st.error("Дата 'с' не может быть больше даты 'по'.")
        else:
            with st.spinner("Запрашиваем данные MOEX..."):
                try:
                    df_index = fetch_index_weights(
                        request_get=request_get,
                        index_name=index_code,
                        date_from=date_from.strftime("%Y-%m-%d"),
                        date_to=date_to.strftime("%Y-%m-%d"),
                    )
                    st.session_state["index_matrix_df"] = df_index
                    st.session_state["index_weight_matrix"] = None
                    st.success(f"Загружено строк: {len(df_index)}")
                except Exception as exc:
                    st.error(f"Ошибка загрузки индекса: {exc}")

    current_df = st.session_state.get("index_matrix_df")
    if current_df is None:
        return

    st.markdown("#### Таблица состава")
    st.dataframe(current_df, use_container_width=True)

    st.download_button(
        label="💾 Скачать таблицу (CSV)",
        data=current_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"index_weights_{index_code.upper()}.csv",
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
        file_name=f"index_weight_matrix_{index_code.upper()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="index_matrix_xlsx_dl",
    )
