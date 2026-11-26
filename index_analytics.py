from datetime import datetime, timedelta
from io import BytesIO
import re
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

from sell_stress_ui.data import fetch_historical_index_memberships_for_isins, fetch_index_membership_for_isin
from sell_stress_ui.reporting import INDEX_CATALOG


INDEX_ANALYTICS_HELP_TEXT = ""


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




def _parse_manual_dates(raw_dates: str) -> List[pd.Timestamp]:
    if not raw_dates or not raw_dates.strip():
        raise ValueError("Укажите хотя бы одну дату в списке.")

    normalized_tokens = raw_dates.replace(";", "\n").replace(",", "\n")
    dates: List[pd.Timestamp] = []
    invalid_tokens: List[str] = []

    for token in normalized_tokens.splitlines():
        value = token.strip()
        if not value:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            invalid_tokens.append(value)
            continue
        dates.append(parsed.normalize())

    if invalid_tokens:
        raise ValueError(
            "Не удалось распознать даты: " + ", ".join(invalid_tokens[:5])
        )

    if not dates:
        raise ValueError("Укажите хотя бы одну корректную дату.")

    unique_dates = sorted(set(dates))
    return unique_dates


def _collect_index_snapshots(
    request_get,
    index_name: str,
    requested_dates: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    collected: List[pd.DataFrame] = []
    seen_dates: Set[pd.Timestamp] = set()

    for requested_dt in sorted(set(requested_dates)):
        snapshot_df, _ = _fetch_index_snapshot(
            index_name=index_name,
            date_str=requested_dt.strftime("%Y-%m-%d"),
            _request_get_func=request_get,
        )
        if snapshot_df.empty:
            continue

        effective_dt = snapshot_df["Date"].max().normalize()
        if effective_dt in seen_dates:
            continue

        day_df = snapshot_df[snapshot_df["Date"].dt.normalize() == effective_dt]
        if day_df.empty:
            continue

        collected.append(day_df)
        seen_dates.add(effective_dt)

    if not collected:
        return _empty_index_weights_df()

    return pd.concat(collected, ignore_index=True)

def fetch_index_weights(
    request_get,
    index_name: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    requested_dates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    index_name = str(index_name).strip().upper()
    if not index_name:
        raise ValueError("Укажите код индекса, например IMOEX")

    if requested_dates is not None:
        parsed_dates = [pd.to_datetime(date_value, errors="coerce") for date_value in requested_dates]
        if any(pd.isna(dt) for dt in parsed_dates):
            raise ValueError("В списке дат есть некорректные значения")

        normalized_dates = [dt.normalize() for dt in parsed_dates]
        df = _collect_index_snapshots(request_get, index_name, normalized_dates)
    else:
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

    st.subheader("🧾 Состав индекса")
    st.markdown(INDEX_ANALYTICS_HELP_TEXT)

    st.markdown("#### Историческая выгрузка составов индексов по нескольким ISIN")
    st.caption(
        "Введите один или несколько ISIN и выберите дату. В Excel будут отдельные листы "
        "с детализацией по индексам и со сводкой по каждому ISIN."
    )
    with st.form("historical_index_membership_export_form"):
        historical_isins_raw = st.text_area(
            "ISIN (по одному на строку, также можно разделять запятыми или точкой с запятой)",
            placeholder="RU0009029540\nRU000A0JX0J2",
            height=110,
            key="historical_index_membership_isins",
        )
        historical_as_of = st.date_input(
            "Дата состава индексов",
            value=datetime.today().date(),
            key="historical_index_membership_date",
        )
        historical_submitted = st.form_submit_button(
            "Загрузить составы индексов",
            use_container_width=True,
        )

    if historical_submitted:
        raw_tokens = re.split(r"[\\s,;]+", historical_isins_raw.strip().upper())
        historical_isins = list(dict.fromkeys(token for token in raw_tokens if token))
        invalid_isins = [value for value in historical_isins if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", value)]
        st.session_state["historical_index_membership_detail"] = None
        if not historical_isins:
            st.warning("Введите хотя бы один ISIN.")
        elif invalid_isins:
            st.error("Проверьте формат ISIN (12 символов): " + ", ".join(invalid_isins))
        else:
            with st.spinner("Загружаем исторические составы индексов MOEX..."):
                try:
                    detail = fetch_historical_index_memberships_for_isins(
                        historical_isins,
                        historical_as_of.strftime("%Y-%m-%d"),
                    )
                    st.session_state["historical_index_membership_detail"] = detail
                    st.session_state["historical_index_membership_date_value"] = historical_as_of
                except Exception as exc:
                    st.error(f"Ошибка загрузки исторических составов: {exc}")

    historical_detail = st.session_state.get("historical_index_membership_detail")
    if historical_detail is not None:
        historical_failed_indices = historical_detail.attrs.get("failed_indices", [])
        historical_failed_isins = historical_detail.attrs.get("failed_isins", [])
        index_names = {item["code"]: item["name"] for item in INDEX_CATALOG}
        detail_view = historical_detail.copy()
        if not detail_view.empty:
            detail_view.insert(
                5,
                "Index name",
                detail_view["Index code"].map(index_names).fillna(detail_view["Index code"]),
            )
            detail_view = detail_view[
                ["Requested date", "Snapshot date", "ISIN", "Ticker", "Index name", "Index code", "Weight"]
            ]

        if detail_view.empty:
            summary_view = pd.DataFrame(columns=[
                "ISIN", "Ticker", "Index count", "Index codes", "Index names", "Snapshot dates"
            ])
            st.info("На выбранную дату в проверенных составах индексов совпадений не найдено.")
        else:
            summary_view = detail_view.groupby(["ISIN", "Ticker"], as_index=False).agg(
                **{
                    "Index count": ("Index code", "nunique"),
                    "Index codes": ("Index code", lambda values: ", ".join(sorted(set(values)))),
                    "Index names": ("Index name", lambda values: "; ".join(dict.fromkeys(values))),
                    "Snapshot dates": ("Snapshot date", lambda values: ", ".join(sorted({str(v) for v in values}))),
                }
            )
            st.success(
                f"Найдено включений в индексы: {len(detail_view)}; "
                f"ISIN с найденными вхождениями: {summary_view['ISIN'].nunique()}."
            )
            st.markdown("**Подробная таблица: ISIN × индекс**")
            st.dataframe(detail_view, use_container_width=True, hide_index=True)

        requested_isins = historical_detail.attrs.get("requested_isins", [])
        requested_frame = pd.DataFrame({"ISIN": requested_isins})
        summary_view = requested_frame.merge(summary_view, on="ISIN", how="left")
        for column, default in [
            ("Ticker", ""), ("Index count", 0), ("Index codes", ""),
            ("Index names", ""), ("Snapshot dates", ""),
        ]:
            if column not in summary_view.columns:
                summary_view[column] = default
            summary_view[column] = summary_view[column].fillna(default)
        summary_view["Index count"] = summary_view["Index count"].astype(int)
        summary_view = summary_view[
            ["ISIN", "Ticker", "Index count", "Index codes", "Index names", "Snapshot dates"]
        ].sort_values(["Index count", "ISIN"], ascending=[False, True])

        st.markdown("**Сводка по каждому введённому ISIN**")
        st.dataframe(summary_view, use_container_width=True, hide_index=True)

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            detail_view.to_excel(writer, sheet_name="Состав индексов", index=False)
            summary_view.to_excel(writer, sheet_name="Сводка по ISIN", index=False)
        export_date = st.session_state.get("historical_index_membership_date_value")
        export_date_str = export_date.strftime("%Y%m%d") if export_date else "date"
        st.download_button(
            "💾 Скачать Excel (2 листа)",
            data=excel_buffer.getvalue(),
            file_name=f"moex_index_membership_{export_date_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="historical_index_membership_xlsx",
        )
        if historical_failed_indices:
            st.warning(
                "Не удалось проверить часть индексов: " + ", ".join(historical_failed_indices)
                + ". Результат может быть неполным."
            )
        if historical_failed_isins:
            st.warning(
                "Не удалось разрешить часть ISIN через MOEX ISS: "
                + ", ".join(historical_failed_isins)
            )

    st.divider()

    st.markdown("#### Поиск всех индексов по ISIN")
    st.caption(
        "Проверяется актуальный состав основного, отраслевых и тематических "
        "индексов акций MOEX. Для каждого найденного индекса показывается вес бумаги."
    )
    with st.form("index_membership_by_isin_form"):
        isin_query = st.text_input(
            "Введите ISIN бумаги",
            placeholder="Например, RU0009029540",
            key="index_membership_isin_input",
        ).strip().upper()
        search_isin_submitted = st.form_submit_button(
            "Найти все индексы",
            use_container_width=True,
        )

    if search_isin_submitted:
        st.session_state["index_isin_membership_result"] = None
        st.session_state["index_isin_membership_query"] = isin_query
        if not isin_query:
            st.warning("Введите ISIN для поиска.")
        elif not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin_query):
            st.error("Проверьте формат ISIN: ожидается 12 символов.")
        else:
            with st.spinner("Проверяем состав всех индексов MOEX..."):
                try:
                    membership = fetch_index_membership_for_isin(isin_query)
                    st.session_state["index_isin_membership_result"] = membership
                    st.session_state["index_isin_membership_query"] = isin_query
                except Exception as exc:
                    st.session_state["index_isin_membership_result"] = None
                    st.error(f"Ошибка поиска индексов по ISIN: {exc}")

    membership_result = st.session_state.get("index_isin_membership_result")
    searched_isin = st.session_state.get("index_isin_membership_query", "")
    if membership_result is not None:
        failed_indices = membership_result.attrs.get("failed_indices", [])
        if membership_result.empty:
            if failed_indices:
                st.warning(
                    "Не удалось проверить часть индексов: "
                    + ", ".join(failed_indices)
                    + ". Отсутствие бумаги в индексах пока нельзя подтвердить."
                )
            else:
                st.info(f"Бумага {searched_isin} не найдена в проверенных составах индексов MOEX.")
        else:
            index_names = {item["code"]: item["name"] for item in INDEX_CATALOG}
            membership_view = membership_result.copy()
            membership_view.insert(
                1,
                "Index name",
                membership_view["Index code"].map(index_names).fillna(membership_view["Index code"]),
            )
            st.success(
                f"ISIN {searched_isin}: найдено индексов — "
                f"{membership_view['Index code'].nunique()}."
            )
            st.dataframe(membership_view, use_container_width=True, hide_index=True)
            if failed_indices:
                st.warning("Не удалось проверить индексы: " + ", ".join(failed_indices))
            st.download_button(
                "💾 Скачать список индексов (CSV)",
                data=membership_view.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"moex_indices_{searched_isin}.csv",
                mime="text/csv",
                key="index_isin_membership_csv",
            )

    st.divider()

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
        load_mode = st.radio(
            "Режим загрузки",
            options=["Одна дата", "Период", "Список дат"],
            index=0,
            horizontal=False,
            key="idx_load_mode",
            help="Можно загрузить одну дату, диапазон дат или список дат, введенный вручную.",
        )

    st.caption("Если код индекса не указан, используется IMOEX.")

    date_from = None
    date_to = None
    manual_dates_raw = ""

    if load_mode == "Период":
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
    elif load_mode == "Список дат":
        manual_dates_raw = st.text_area(
            "Список дат",
            value="",
            height=140,
            placeholder="2026-03-01\n2026-03-05\n2026-03-10",
            help="Вносите даты вручную: по одной на строку, либо через запятую/точку с запятой.",
            key="idx_manual_dates",
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
        try:
            manual_dates = _parse_manual_dates(manual_dates_raw) if load_mode == "Список дат" else None
        except ValueError as exc:
            st.error(str(exc))
            manual_dates = None

        if load_mode == "Период" and date_from > date_to:
            st.error("Дата 'с' не может быть больше даты 'по'.")
        elif load_mode == "Список дат" and not manual_dates:
            pass
        else:
            with st.spinner("Формируется..."):
                try:
                    df_index = fetch_index_weights(
                        request_get=request_get,
                        index_name=index_code,
                        date_from=date_from.strftime("%Y-%m-%d") if date_from else None,
                        date_to=date_to.strftime("%Y-%m-%d") if date_to else None,
                        requested_dates=[dt.strftime("%Y-%m-%d") for dt in manual_dates] if manual_dates else None,
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

    if st.button("Сформировать таблицу именения весов", key="build_index_weight_matrix"):
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

