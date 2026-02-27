import csv
import math
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO, StringIO
from urllib.parse import quote, urlencode
from email.message import EmailMessage

import numpy as np
import pandas as pd
import requests
import sell_stress as ss
import streamlit as st
import index_analytics as ia
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


if st.session_state["active_view"] != "home":
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
    with bottom_right:
        st.markdown("### Sell_stress")
        st.caption("Оценка рыночного давления для акций и облигаций.")
        if st.button("Открыть", key="open_sell_stres", use_container_width=True):
            st.session_state["active_view"] = "sell_stres"
            trigger_rerun()
    index_col, _ = st.columns(2)
    with index_col:
        st.markdown("### Состав индекса")
        st.caption("Загрузка состава индекса по датам и построение матрицы весов.")
        if st.button("Открыть", key="open_index_analytics", use_container_width=True):
            st.session_state["active_view"] = "index_analytics"
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


def open_index_analytics_sheet():
    ia.render_index_analytics_view(
        request_get=request_get,
        dataframe_to_excel_bytes=ss.dataframe_to_excel_bytes,
    )


def parse_email_list(raw_recipients: str) -> tuple[list[str], list[str]]:
    chunks = re.split(r"[;,\s]+", raw_recipients.strip()) if raw_recipients else []
    unique = []
    seen = set()
    email_pattern = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    invalid = []
    for item in chunks:
        email = item.strip()
        if not email:
            continue
        if email in seen:
            continue
        seen.add(email)
        if email_pattern.match(email):
            unique.append(email)
        else:
            invalid.append(email)
    return unique, invalid


def build_compose_link(service: str, recipients: list[str], cc_recipients: list[str], subject: str, body: str) -> str:
    query_string = lambda params: urlencode({k: v for k, v in params.items() if v}, quote_via=quote)
    to_field = ",".join(recipients)
    cc_field = ",".join(cc_recipients)
    if service == "Почтовый клиент по умолчанию":
        mailto_params = query_string({"cc": cc_field, "subject": subject, "body": body})
        return f"mailto:{to_field}" + (f"?{mailto_params}" if mailto_params else "")
    if service == "Gmail":
        return "https://mail.google.com/mail/?" + query_string(
            {"view": "cm", "fs": "1", "to": to_field, "cc": cc_field, "su": subject, "body": body}
        )
    if service == "Outlook Web":
        return "https://outlook.office.com/mail/deeplink/compose?" + query_string(
            {"to": to_field, "cc": cc_field, "subject": subject, "body": body}
        )
    if service == "Yandex Mail":
        return "https://mail.yandex.ru/compose?" + query_string(
            {"to": to_field, "cc": cc_field, "subject": subject, "body": body}
        )
    return "https://e.mail.ru/compose/?" + query_string(
        {"To": to_field, "Cc": cc_field, "Subject": subject, "Body": body}
    )


def build_eml_attachment(
    recipients: list[str],
    cc_recipients: list[str],
    subject: str,
    body: str,
    attachment_name: str,
    attachment_bytes: bytes,
) -> bytes:
    message = EmailMessage()
    message["To"] = ", ".join(recipients)
    if cc_recipients:
        message["Cc"] = ", ".join(cc_recipients)
    message["Subject"] = subject
    message.set_content(body)
    message.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=attachment_name,
    )
    return message.as_bytes()


def render_email_compose_section(
    report_title: str,
    key_prefix: str,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
):
    st.markdown("---")
    st.subheader("📧 Отправка отчёта по почте")
    st.caption(
        "Выберите сервис и адреса — приложение откроет черновик письма. "
        "Можно добавить адреса в копию и скачать .eml с Excel-вложением. "
        "Подпись/бланк подставляется почтовым клиентом по вашим настройкам."
    )

    mail_service = st.selectbox(
        "Почтовый сервис",
        ["Почтовый клиент по умолчанию", "Gmail", "Outlook Web", "Yandex Mail", "Mail.ru"],
        key=f"{key_prefix}_service",
    )
    recipients_raw = st.text_area(
        "Адреса получателей (через запятую, точку с запятой или перенос строки)",
        placeholder="user1@example.com; user2@example.com",
        key=f"{key_prefix}_recipients",
    )
    cc_recipients_raw = st.text_area(
        "Адреса в копии (CC)",
        placeholder="copy1@example.com; copy2@example.com",
        key=f"{key_prefix}_cc_recipients",
    )
    default_subject = f"{report_title} на {datetime.today().strftime('%d.%m.%Y')}"
    mail_subject = st.text_input("Тема письма", value=default_subject, key=f"{key_prefix}_subject")
    mail_body = st.text_area(
        "Текст письма",
        value=(
            "Коллеги, добрый день!\n\n"
            f"Направляю {report_title.lower()}.\n"
            "Пожалуйста, см. вложенный файл.\n\n"
        ),
        height=180,
        key=f"{key_prefix}_body",
    )

    if st.button("Сгенерировать письмо", key=f"{key_prefix}_generate"):
        recipients, invalid_recipients = parse_email_list(recipients_raw)
        cc_recipients, invalid_cc_recipients = parse_email_list(cc_recipients_raw)
        invalid_emails = invalid_recipients + invalid_cc_recipients
        if invalid_emails:
            st.error(
                "Некорректные адреса: "
                + ", ".join(invalid_emails[:10])
                + ("..." if len(invalid_emails) > 10 else "")
            )
        if not recipients:
            st.warning("Укажите хотя бы один корректный email получателя.")
        if recipients:
            subject = mail_subject.strip()
            body = mail_body.strip()
            compose_link = build_compose_link(mail_service, recipients, cc_recipients, subject, body)
            st.success(f"Черновик подготовлен для {len(recipients)} получателя(ей).")
            st.link_button("Открыть письмо в выбранном сервисе", compose_link)
            st.code(compose_link, language="text")
            if attachment_name and attachment_bytes:
                eml_bytes = build_eml_attachment(
                    recipients,
                    cc_recipients,
                    subject,
                    body,
                    attachment_name,
                    attachment_bytes,
                )
                st.download_button(
                    "📎 Скачать черновик .eml с Excel-вложением",
                    data=eml_bytes,
                    file_name="report_with_attachment.eml",
                    mime="message/rfc822",
                    key=f"{key_prefix}_eml_download",
                )


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
        "iss.only": "securities",
        "securities.columns": "PREVSETTLEPRICE,MINSTEP,STEPPRICE,LASTSETTLEPRICE",
    }
    spec = request_get(spec_url, timeout=2000, params=spec_params).json()
    prev_settle_raw, minstep_raw, stepprice_raw, last_settle_raw = spec["securities"]["data"][0]
    prev_settle = to_decimal(prev_settle_raw)
    minstep = to_decimal(minstep_raw)
    stepprice = to_decimal(stepprice_raw)
    last_settle = to_decimal(last_settle_raw) if last_settle_raw is not None else None

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
    vm = (day_settle - prev_settle) * multiplier

    return {
        "TRADE_NAME": trade_name_clean,
        "SECID": secid,
        "TRADEDATE": trade_date,
        "PREV_PRICE": float(money_decimal(prev_settle)),
        "LAST_SETTLE_PRICE": float(money_decimal(last_settle)) if last_settle is not None else None,
        "TODAY_PRICE": float(money_decimal(day_settle)),
        "MULTIPLIER": float(multiplier),
        "VM": float(money_decimal(vm)),
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
            st.dataframe(df_timeline, use_container_width=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_timeline.to_excel(writer, sheet_name="calendar")
            calendar_xlsx = output.getvalue()
            calendar_csv = df_timeline.to_csv(index=True).encode("utf-8-sig")

            st.download_button(
                label="💾 Скачать календарь (Excel)",
                data=calendar_xlsx,
                file_name="bond_calendar.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="calendar_xlsx_dl",
            )
            st.download_button(
                label="💾 Скачать календарь (CSV)",
                data=calendar_csv,
                file_name="bond_calendar.csv",
                mime="text/csv",
                key="calendar_csv_dl",
            )

            render_email_compose_section("Календарь выплат", "calendar_report", "bond_calendar.xlsx", calendar_xlsx)
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
                limit_sum = (0.05 * vm_data["TODAY_PRICE"] * quantity * usd_rub) + position_vm
                st.session_state["vm_last_report"] = {
                    "TRADE_NAME": vm_data["TRADE_NAME"],
                    "SECID": vm_data["SECID"],
                    "TRADEDATE": vm_data["TRADEDATE"],
                    "LAST_SETTLE_PRICE": vm_data["LAST_SETTLE_PRICE"],
                    "TODAY_PRICE": vm_data["TODAY_PRICE"],
                    "MULTIPLIER": vm_data["MULTIPLIER"],
                    "VM": vm_data["VM"],
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
        st.markdown(f"**Последняя цена:** {vm_report['TODAY_PRICE']}")
        st.markdown(f"**Multiplier:** {vm_report['MULTIPLIER']}")
        st.markdown(f"**Вариационная маржа за день:** {vm_report['VM']:.2f}")
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

        render_email_compose_section("VM отчёт", "vm_report", "vm_report.xlsx", vm_xlsx)

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
            "Вводить Q для каждого ISIN (формат: ISIN | Q)", value=False, key="share_q_per_isin"
        )
        if use_q_from_list:
            isin_q_input = st.text_area(
                "Введите ISIN и Q (каждая строка: ISIN | Q)",
                height=160,
                placeholder="RU0009029540 | 33000000000\nRU000A0JX0J2 | 25000000000",
                key="share_isin_q_input",
            )
        else:
            isin_input = st.text_area(
                "Введите или вставьте ISIN (через Ctrl+V, пробел или запятую)",
                height=160,
                placeholder="RU0009029540\nRU000A0JX0J2",
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
            invalid_isins = []
            if use_q_from_list:
                raw_lines = [line.strip() for line in isin_q_input.splitlines() if line.strip()]
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
                raw_text = isin_input.strip()
                if raw_text:
                    isins = re.split(r"[\s,;]+", raw_text)
                    isins = [i.strip().upper() for i in isins if i.strip()]
                    for isin in isins:
                        if not isin_format_valid(isin):
                            invalid_isins.append(isin)
                            continue
                        entries.append({"ISIN": isin, "Q_MAX": int(q_max)})

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
