import csv
import asyncio
import os
import math
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO, StringIO
from pathlib import Path

import altair as alt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
import sell_stress as ss
import streamlit as st
import index_analytics as ia
from sell_stress_ui.data import ALL_STOCK_INDEX_CODES, fetch_index_membership_by_isin
from sell_stress_ui.reporting import build_share_batch_html_report
from email_compose import render_email_compose_section
from news.fetcher import NewsFetcher
from news.models import NewsQuery
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from services.moex_turnover import MoexTurnoverClient
from services.company_news_analysis import get_company_news_analysis_sync
from services.news_service import NewsServiceError, get_news, get_news_by_date, get_news_by_isin
from services.keyword_news_block import build_keyword_news_block_sync

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
    "moex_news",
    "company_analysis",
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


@st.cache_data(show_spinner=False)
def get_sell_stress_xml_form_config():
    xml_path = Path(__file__).resolve().parent / "sell_stress_ui" / "schemas" / "sell_stress_form.xml"
    return load_sell_stress_form_config(xml_path)


@st.cache_data(show_spinner=False)
def get_sell_stress_asset_universe():
    return load_asset_universe()


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
    st.markdown("### AI Анализ компании")
    st.caption("Новости, инвестиционный сигнал и факторная расшифровка в отдельной плитке.")
    if st.button("Открыть", key="open_company_analysis_tile", use_container_width=True):
        st.session_state["active_view"] = "company_analysis"
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

        st.markdown("### Новости MOEX")
        st.caption("Поиск новостей MOEX по дате и ISIN с подбором связанных публикаций эмитента.")
        if st.button("Открыть", key="open_moex_news", use_container_width=True):
            st.session_state["active_view"] = "moex_news"
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
        st.markdown("### Выгрузка оборотов")
        st.caption("Обороты акций/облигаций за период с опцией NDM и Excel-отчётом.")
        if st.button("Открыть", key="open_turnover_export", use_container_width=True):
            st.session_state["active_view"] = "turnover_export"
            trigger_rerun()
    st.stop()

# ---------------------------
# HTTP session with retries
# ---------------------------
GRAMS_PER_TROY_OUNCE = 0.03574


def convert_ounce_price_to_gram(price_series):
    return price_series / GRAMS_PER_TROY_OUNCE


def format_int_with_sep(value):
    rounded = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"{rounded:,}".replace(",", " ")


def safe_format_int_with_sep(value):
    formatter = globals().get("format_int_with_sep")
    if callable(formatter):
        return formatter(value)
    rounded = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"{rounded:,}".replace(",", " ")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_gold_chart_data():
    def _download(symbol, period, interval, prepost=False):
        try:
            return yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                prepost=prepost,
            )
        except Exception:
            return pd.DataFrame()

    daily_candidates = [
        ("GC=F", "6mo", "1d"),
        ("XAUUSD=X", "6mo", "1d"),
        ("GC=F", "1y", "1d"),
        ("XAUUSD=X", "1y", "1d"),
    ]
    intraday_candidates = [
        ("GC=F", "1d", "1m"),
        ("XAUUSD=X", "5d", "5m"),
        ("GC=F", "5d", "5m"),
    ]

    daily = pd.DataFrame()
    for symbol, period, interval in daily_candidates:
        daily = _download(symbol, period, interval)
        if daily is not None and not daily.empty:
            break

    intraday = pd.DataFrame()
    for symbol, period, interval in intraday_candidates:
        intraday = _download(symbol, period, interval, prepost=False)
        if intraday is not None and not intraday.empty:
            break

    return daily, intraday


def _normalize_close_series(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series.dropna()


def get_gold_close_series():
    daily_raw, intraday_raw = fetch_gold_chart_data()
    daily_close = convert_ounce_price_to_gram(_normalize_close_series(daily_raw))
    intraday_close = convert_ounce_price_to_gram(_normalize_close_series(intraday_raw))
    return daily_close, intraday_close


def _extract_news_entries(payload, bucket):
    if isinstance(payload, dict):
        has_title = isinstance(payload.get("title"), str)
        has_url = isinstance(payload.get("url"), str)
        published = payload.get("publishedAt") or payload.get("datePublished") or payload.get("published")
        if has_title and has_url and isinstance(published, str):
            bucket.append(
                {
                    "title": payload.get("title", "").strip(),
                    "url": payload.get("url", "").strip(),
                    "published_at": str(published).strip(),
                    "source": str(payload.get("publisher", "")) or "TradingView",
                }
            )
        for value in payload.values():
            _extract_news_entries(value, bucket)
    elif isinstance(payload, list):
        for item in payload:
            _extract_news_entries(item, bucket)


def _parse_news_datetime(value: str):
    if not value:
        return None
    value = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            return dt.astimezone(tz=None).replace(tzinfo=None)
        return dt
    except ValueError:
        pass
    for fmt in ("%a, %d %b %Y %H:%M:%S GMT", "%b %d, %Y, %H:%M %Z"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt
        except ValueError:
            continue
    return None


def _strip_html_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def _extract_news_from_cards(html: str):
    items = []
    card_pattern = re.compile(
        r'<a[^>]+href="(?P<href>/news/[^"]+)"[^>]*>.*?'
        r'<relative-time[^>]+event-time="(?P<event_time>[^"]+)"[^>]*>.*?</relative-time>.*?'
        r'<span class="provider-[^"]*"><span[^>]*>(?P<provider>.*?)</span>.*?</span>.*?'
        r'<div[^>]+data-qa-id="news-headline-title"[^>]*>(?P<title>.*?)</div>.*?'
        r"</a>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for match in card_pattern.finditer(html):
        href = match.group("href") or ""
        title = _strip_html_tags(match.group("title"))
        provider = _strip_html_tags(match.group("provider"))
        event_time = (match.group("event_time") or "").strip()
        if not title or not event_time:
            continue
        items.append(
            {
                "title": title,
                "url": f"https://www.tradingview.com{href}" if href.startswith("/") else href,
                "published_at": event_time,
                "source": provider or "TradingView",
            }
        )
    return items


def _fetch_xauusd_news_like_ai_analysis():
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=1)
    query = NewsQuery(
        query="XAUUSD OR Gold spot OR Gold price",
        start_date=start_utc,
        end_date=now_utc,
        language="en",
        limit=40,
    )
    try:
        fetched_news = asyncio.run(NewsFetcher().fetch_news(query))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            fetched_news = loop.run_until_complete(NewsFetcher().fetch_news(query))
        finally:
            loop.close()
    prepared = []
    for item in fetched_news:
        published = item.published_at
        if published.tzinfo is not None:
            published = published.astimezone(timezone.utc).replace(tzinfo=None)
        if published >= datetime.utcnow() - timedelta(days=1):
            prepared.append(
                {
                    "title": item.title.strip(),
                    "url": item.url.strip(),
                    "published_at": published.strftime("%Y-%m-%d %H:%M"),
                }
            )
    prepared.sort(key=lambda x: x["published_at"], reverse=True)
    if not prepared:
        for item in fetched_news[:15]:
            published = item.published_at
            if published.tzinfo is not None:
                published = published.astimezone(timezone.utc).replace(tzinfo=None)
            prepared.append(
                {
                    "title": item.title.strip(),
                    "url": item.url.strip(),
                    "published_at": published.strftime("%Y-%m-%d %H:%M"),
                }
            )
    return prepared


@st.cache_data(ttl=900, show_spinner=False)
def fetch_xauusd_tradingview_news():
    url = "https://www.tradingview.com/symbols/XAUUSD/news/?exchange=OANDA"
    response = HTTP_SESSION.get(
        url,
        timeout=20,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    html = response.text
    scripts = re.findall(
        r'<script[^>]*type="application/ld\\+json"[^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not scripts:
        scripts = re.findall(
            r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

    parsed_entries = []
    for block in scripts:
        try:
            payload = json.loads(block.strip())
        except Exception:
            continue
        _extract_news_entries(payload, parsed_entries)
    parsed_entries.extend(_extract_news_from_cards(html))

    if not parsed_entries:
        return _fetch_xauusd_news_like_ai_analysis()

    unique_news = {}
    for item in parsed_entries:
        news_url = item.get("url")
        if news_url and news_url not in unique_news:
            unique_news[news_url] = item

    yesterday_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    filtered = []
    for item in unique_news.values():
        parsed_dt = _parse_news_datetime(item.get("published_at", ""))
        if parsed_dt and parsed_dt >= yesterday_start:
            filtered.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "published_at": parsed_dt.strftime("%Y-%m-%d %H:%M"),
                }
            )

    filtered.sort(key=lambda x: x["published_at"], reverse=True)
    if not filtered:
        return _fetch_xauusd_news_like_ai_analysis()
    return filtered


def calculate_var_results(result_D, K_values=None, t=0.95):
    result_D = np.asarray(result_D, dtype=float)
    result_D = result_D[np.isfinite(result_D)]

    if result_D.size == 0:
        return {
            "confidence_level": t,
            "T": norm.ppf(t),
            "Q": None,
            "result_D": result_D,
            "K_values": K_values or [],
            "VAR_results": {},
        }

    if K_values is None:
        K_values = [1, 5, 10]

    T = norm.ppf(t)
    Q = float(np.std(result_D))
    var_results = {}
    for K in K_values:
        M = float(np.mean(result_D) * K)
        var_percent = (-Q * np.sqrt(K) * T + M) * 100
        var_results[int(K)] = float(var_percent)

    return {
        "confidence_level": t,
        "T": float(T),
        "Q": Q,
        "result_D": result_D,
        "K_values": [int(k) for k in K_values],
        "VAR_results": var_results,
    }


def build_var_table(var_payload):
    var_results = var_payload.get("VAR_results", {})
    if not var_results:
        return pd.DataFrame(columns=["Дни", "VaR, %"])
    return pd.DataFrame(
        [{"Дни": int(k), "VaR, %": float(v)} for k, v in var_results.items()]
    )


def _style_gold_axis(ax, title, xlabel, ylabel, formatter=None, y_formatter=None):
    ax.set_title(title, fontsize=12, fontweight="normal", color="#262730", pad=10)
    ax.set_xlabel(xlabel, color="#262730")
    ax.set_ylabel(ylabel, color="#262730")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.18, color="#d9d9d9")
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d9d9d9")
    ax.spines["bottom"].set_color("#d9d9d9")
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="x", labelrotation=25, colors="#262730")
    ax.tick_params(axis="y", colors="#262730")


def get_gold_y_bounds(series, intraday=False):
    min_value = float(series.min())
    max_value = float(series.max())
    if min_value == max_value:
        padding = abs(max_value) * 0.1 if max_value else 1.0
        return min_value - padding, max_value + padding
    spread = max_value - min_value
    dynamic_padding = spread * (0.12 if intraday else 0.10)
    min_floor_padding = max(abs(max_value), abs(min_value)) * 0.002
    padding = max(dynamic_padding, min_floor_padding)
    lower_bound = min_value - padding
    upper_bound = max_value + padding
    return lower_bound, upper_bound


def _apply_gold_y_padding(ax, series, intraday=False):
    lower_bound, upper_bound = get_gold_y_bounds(series, intraday=intraday)
    ax.set_ylim(lower_bound, upper_bound)



def build_gold_chart_display(series, title, intraday=False):
    df = series.reset_index()
    x_col = df.columns[0]
    y_col = df.columns[1]
    lower_bound, upper_bound = get_gold_y_bounds(series)
    axis_format = "%H:%M" if intraday else "%d.%m.%Y"
    chart = (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X(x_col, title="Date / Time", axis=alt.Axis(format=axis_format, labelAngle=-25)),
            y=alt.Y(y_col, title="Price per gram", scale=alt.Scale(domain=[lower_bound, upper_bound])),
            tooltip=[alt.Tooltip(x_col, title="Date / Time"), alt.Tooltip(y_col, title="Price per gram", format=",.4f")],
        )
        .properties(title=title, height=320)
        .interactive()
    )
    return chart



def build_gold_chart_figure(series, title, color, fill_color, intraday=False):
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#ffffff")
    ax.plot(series.index, series.values, color=color, linewidth=2.0)
    _apply_gold_y_padding(ax, series, intraday=intraday)
    formatter = mdates.DateFormatter("%H:%M") if intraday else mdates.DateFormatter("%d.%m.%Y")
    thousands_formatter = FuncFormatter(lambda value, _: f"{value / 1000:.1f}")
    _style_gold_axis(
        ax,
        title,
        "Date / Time",
        "Price per gram, thousand",
        formatter=formatter,
        y_formatter=thousands_formatter,
    )
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    return buffer.getvalue()


def get_gold_var_payload(K_values=None, t=0.95):
    gold_close_series, _ = get_gold_close_series()
    result_D = gold_close_series.pct_change().dropna().to_numpy()
    var_payload = calculate_var_results(result_D, K_values=K_values, t=t)
    var_payload["result_D"] = result_D.tolist()
    return var_payload


def render_gold_charts():
    daily_close, intraday_close = get_gold_close_series()

    if daily_close.empty and intraday_close.empty:
        st.info("Не удалось загрузить данные по золоту из yfinance для построения графиков.")
        return

    chart_columns = st.columns(2)

    with chart_columns[0]:
        st.markdown("#### Gold Daily Close (6M) - per gram")
        if daily_close.empty:
            st.info("Дневные данные по золоту за последние 6 месяцев временно недоступны.")
        else:
            chart = build_gold_chart_display(
                daily_close,
                "Gold Daily Close (6M) - per gram",
            )
            st.altair_chart(chart, use_container_width=True)

    with chart_columns[1]:
        st.markdown("#### Gold Intraday (1M) - per gram")
        if intraday_close.empty:
            st.info("Нет внутридневных данных по золоту за текущий день.")
        else:
            chart = build_gold_chart_display(
                intraday_close,
                "Gold Intraday (1M) - per gram",
                intraday=True,
            )
            st.altair_chart(chart, use_container_width=True)


def get_intraday_chart_attachment():
    _, intraday_close = get_gold_close_series()
    if intraday_close.empty:
        return None
    fig = build_gold_chart_figure(
        intraday_close,
        "Gold Intraday (1M) - per gram",
        color="#1f77b4",
        fill_color="#1f77b4",
        intraday=True,
    )
    png_bytes = figure_to_png_bytes(fig)
    plt.close(fig)
    return ("gold_intraday_1m.png", png_bytes, "image", "png")


def build_vm_pdf_report(vm_report):
    daily_close, intraday_close = get_gold_close_series()
    pdf_buffer = BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig_cover = plt.figure(figsize=(11.69, 8.27), facecolor="#ffffff")
        fig_cover.suptitle("VM Dashboard", fontsize=20, fontweight="bold", y=0.98, color="#1f3a5f")
        fig_cover.text(
            0.03,
            0.92,
            f"Инструмент: {vm_report['TRADE_NAME']} ({vm_report['SECID']}) | "
            f"Дата клиринга: {vm_report['TRADEDATE']} | Кол-во: {vm_report['QUANTITY']}",
            fontsize=10.5,
            color="#262730",
        )
        news_items = vm_report.get("XAUUSD_NEWS", [])
        if news_items:
            preview_lines = []
            for item in news_items[:2]:
                title = item.get("title", "")[:85]
                preview_lines.append(f"• {item.get('published_at', '')} | {title}")
            fig_cover.text(
                0.03,
                0.875,
                "XAUUSD новости (TradingView, со вчерашнего дня):\n" + "\n".join(preview_lines),
                fontsize=8.6,
                color="#3c4758",
            )
        vm_rows = [
            ("Последняя цена", f"{vm_report.get('LAST_PRICE') if vm_report.get('LAST_PRICE') is not None else vm_report['TODAY_PRICE']:.4f}"),
            ("Дата цены", vm_report.get("PRICE_DATE", "н/д")),
            ("Время цены", vm_report.get("QUOTE_TIME") or "н/д"),
            ("VM", f"{vm_report['VM']:.2f}"),
            ("VM клиринговая", f"{vm_report.get('VM_CLEARING', vm_report['VM']):.2f}"),
            ("Маржа позиции", safe_format_int_with_sep(vm_report["POSITION_VM"])),
            ("Сумма ограничения", safe_format_int_with_sep(vm_report["LIMIT_SUM"])),
            ("USD/RUB", f"{vm_report['USD_RUB']} ({vm_report['USD_RUB_DATE']})"),
        ]
        ax_table = fig_cover.add_axes([0.03, 0.53, 0.44, 0.33])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=[[key, value] for key, value in vm_rows],
            colLabels=["Показатель", "Значение"],
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e8f0fb")
                cell.set_text_props(weight="bold", color="#1f3a5f")
            else:
                cell.set_facecolor("#f8fbff" if row % 2 == 0 else "#ffffff")
        var_results = vm_report.get("VAR_RESULTS", {})
        ax_var_table = fig_cover.add_axes([0.53, 0.53, 0.44, 0.33])
        ax_var_table.axis("off")
        ax_var_table.text(
            0.0,
            1.05,
            "Value at Risk (VaR)",
            fontsize=13,
            fontweight="bold",
            color="#1f3a5f",
            transform=ax_var_table.transAxes,
        )
        if var_results:
            ax_var_table.text(
                0.0,
                0.92,
                f"Доверительный уровень: {vm_report.get('VAR_CONFIDENCE_LEVEL', 0.95):.2f} | "
                f"T={vm_report.get('VAR_T', 0):.4f} | Q={vm_report.get('VAR_Q', 0):.6f}",
                fontsize=10,
                color="#262730",
                transform=ax_var_table.transAxes,
            )
            var_df = build_var_table({"VAR_results": var_results})
            var_table = ax_var_table.table(
                cellText=[[int(row["Дни"]), f"{float(row['VaR, %']):.4f}"] for _, row in var_df.iterrows()],
                colLabels=["Горизонт, дни", "VaR, %"],
                cellLoc="center",
                colLoc="center",
                loc="lower center",
                bbox=[0, 0.15, 1.0, 0.65],
            )
            var_table.auto_set_font_size(False)
            var_table.set_fontsize(10)
            var_table.scale(1, 1.25)
            for (row, col), cell in var_table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#e8f0fb")
                    cell.set_text_props(weight="bold", color="#1f3a5f")
                else:
                    cell.set_facecolor("#f8fbff" if row % 2 == 0 else "#ffffff")
        elif vm_report.get("VAR_ERROR"):
            ax_var_table.text(
                0.0,
                0.72,
                f"VaR недоступен: {vm_report['VAR_ERROR']}",
                fontsize=10,
                color="#9b2c2c",
                transform=ax_var_table.transAxes,
            )
        else:
            ax_var_table.text(
                0.0,
                0.72,
                "Недостаточно дневной истории золота для расчёта VaR.",
                fontsize=10,
                color="#9b2c2c",
                transform=ax_var_table.transAxes,
            )

        if not daily_close.empty:
            ax_daily = fig_cover.add_axes([0.03, 0.08, 0.44, 0.35])
            ax_daily.plot(daily_close.index, daily_close.values, color="#1f77b4", linewidth=1.8)
            _apply_gold_y_padding(ax_daily, daily_close, intraday=False)
            thousands_formatter = FuncFormatter(lambda value, _: f"{value / 1000:.1f}")
            _style_gold_axis(
                ax_daily,
                "Gold Daily Close (6M) - per gram",
                "Date / Time",
                "Price per gram, thousand",
                formatter=mdates.DateFormatter("%d.%m.%Y"),
                y_formatter=thousands_formatter,
            )
        else:
            ax_daily = fig_cover.add_axes([0.03, 0.08, 0.44, 0.35])
            ax_daily.axis("off")
            ax_daily.text(0.5, 0.5, "Дневные данные временно недоступны.", ha="center", va="center", color="#9b2c2c")

        if not intraday_close.empty:
            ax_intraday = fig_cover.add_axes([0.53, 0.08, 0.44, 0.35])
            ax_intraday.plot(intraday_close.index, intraday_close.values, color="#1f77b4", linewidth=1.8)
            _apply_gold_y_padding(ax_intraday, intraday_close, intraday=True)
            thousands_formatter = FuncFormatter(lambda value, _: f"{value / 1000:.1f}")
            _style_gold_axis(
                ax_intraday,
                "Gold Intraday (1M) - per gram",
                "Date / Time",
                "Price per gram, thousand",
                formatter=mdates.DateFormatter("%H:%M"),
                y_formatter=thousands_formatter,
            )
        else:
            ax_intraday = fig_cover.add_axes([0.53, 0.08, 0.44, 0.35])
            ax_intraday.axis("off")
            ax_intraday.text(0.5, 0.5, "Внутридневные данные временно недоступны.", ha="center", va="center", color="#9b2c2c")

        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


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

    direct_rows: list[list[object]] = []
    direct_columns: list[str] = []
    try:
        direct_response = request_get(
            f"https://iss.moex.com/iss/securities/{normalized}.json",
            params={"iss.meta": "off", "iss.only": "securities"},
            timeout=200,
        )
        direct_payload = direct_response.json().get("securities", {})
        direct_rows = direct_payload.get("data", []) or []
        direct_columns = direct_payload.get("columns", []) or []
    except Exception:
        direct_rows = []
        direct_columns = []

    if direct_rows:
        df_direct = pd.DataFrame(direct_rows, columns=direct_columns)
        if "group" in df_direct.columns:
            if market_kind == "shares":
                share_mask = (
                    df_direct["group"].astype(str).str.contains("share|stock|etf", case=False, na=False)
                )
                df_direct = df_direct[share_mask]
            else:
                df_direct = df_direct[df_direct["group"].astype(str).str.contains("bond", case=False, na=False)]
        if not df_direct.empty:
            row = df_direct.iloc[0]
            secid = str(row.get("secid", "")).strip().upper()
            if secid:
                return {
                    "input": identifier,
                    "secid": secid,
                    "isin": str(row.get("isin", "")).strip().upper(),
                    "shortname": str(row.get("shortname", "")).strip(),
                    "emitent_title": str(row.get("emitent_title", "")).strip(),
                }

    if isin_format_valid(normalized):
        try:
            secid_from_isin = isin_to_secid(normalized)
            if secid_from_isin:
                return {
                    "input": identifier,
                    "secid": str(secid_from_isin).strip().upper(),
                    "isin": normalized,
                    "shortname": "",
                    "emitent_title": "",
                }
        except Exception:
            pass

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
            df_filtered = df[df["group"].astype(str).str.contains("share|stock|etf", case=False, na=False)]
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
    boards: list[str] = []
    try:
        boards_response = request_get(
            f"https://iss.moex.com/iss/history/engines/stock/markets/{market_kind}/boards.json",
            params={"iss.meta": "off", "iss.only": "boards", "boards.columns": "boardid,is_traded"},
            timeout=200,
        )
        boards_payload = boards_response.json().get("boards", {})
        boards_df = pd.DataFrame(boards_payload.get("data", []), columns=boards_payload.get("columns", []))
        if not boards_df.empty and "boardid" in boards_df.columns:
            if "is_traded" in boards_df.columns:
                boards_df["is_traded"] = pd.to_numeric(boards_df["is_traded"], errors="coerce").fillna(0)
                boards_df = boards_df[boards_df["is_traded"] > 0]
            boards = [
                str(board).strip().upper()
                for board in boards_df["boardid"].tolist()
                if str(board).strip()
            ]
    except Exception:
        boards = []

    if not boards:
        boards = ["TQBR"] if market_kind == "shares" else ["TQCB"]

    all_rows: list[list[object]] = []
    columns: list[str] = []
    for board in boards:
        start = 0
        while True:
            url = (
                f"https://iss.moex.com/iss/history/engines/stock/markets/{market_kind}/"
                f"boards/{board}/securities.json"
            )
            response = request_get(
                url,
                params={
                    "from": start_date,
                    "till": end_date,
                    "start": start,
                    "iss.only": "history",
                    "iss.meta": "off",
                    "history.columns": "TRADEDATE,VALUE,NUMTRADES,VOLUME,SHORTNAME,SECID,BOARDID",
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
        "history.columns": "TRADEDATE,SETTLEPRICE",
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
if st.session_state["active_view"] == "company_analysis":
    st.header("Новости по keyword + финансовое LLM summary")
    st.caption("Приоритет: русскоязычные источники (РБК, Интерфакс, Ведомости, Коммерсант, ТАСС) за последние 30 дней.")

    with st.form("company_analysis_news_form"):
        keyword_value = st.text_input(
            "Keyword для поиска новостей",
            value=st.session_state.get("company_analysis_query", "AAPL"),
            key="company_analysis_keyword_input",
        )
        depth_days = st.number_input("Глубина поиска, дней", min_value=7, max_value=365, value=30, step=1)
        summary_variants = st.selectbox("Количество вариантов summary", options=[3, 4, 5], index=0)
        run_clicked = st.form_submit_button("Найти новости и собрать summary", use_container_width=True)

    st.session_state["company_analysis_query"] = keyword_value

    if run_clicked:
        user_query = keyword_value.strip()
        if not user_query:
            st.warning("Введите keyword.")
        else:
            with st.spinner("Поиск новостей в интернете и LLM-агрегация..."):
                try:
                    payload = build_keyword_news_block_sync(
                        user_query,
                        depth_days=int(depth_days),
                        summary_variants=int(summary_variants),
                    )
                except Exception as exc:
                    st.error(f"Ошибка выполнения блока: {exc}")
                else:
                    news_pool = payload.get("news_pool", [])
                    errors = payload.get("errors", [])

                    st.subheader("Пул новостей")
                    st.caption(
                        f"Keyword: {payload.get('keyword')} | окно: {payload.get('window_days')} дней | найдено: {payload.get('news_count', len(news_pool))}"
                    )
                    st.caption("Расширенные ключи: " + ", ".join(payload.get("expanded_keywords", [])))
                    st.caption("Приоритетные источники: " + ", ".join(payload.get("priority_sources", [])))
                    st.json(news_pool)

                    if errors:
                        st.warning("Ошибки источников: " + " | ".join(str(e) for e in errors))

                    st.subheader("Короткое summary (LLM)")
                    if payload.get("best_summary"):
                        st.markdown(f"**Лучший вариант (для обучения модели): Вариант {payload.get('best_variant', 1)}**")
                        st.write(payload.get("best_summary", ""))
                        if payload.get("summary_ranking"):
                            st.caption("Ранжирование вариантов: " + ", ".join(
                                f"#{row.get('variant')} (score={row.get('score')})" for row in payload.get("summary_ranking", [])
                            ))
                    with st.expander("Показать остальные варианты summary"):
                        for idx, summary_text in enumerate(payload.get("summaries", []), start=1):
                            st.markdown(f"**Вариант {idx}**")
                            st.write(summary_text)
    st.stop()

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
                price_date = datetime.utcnow().strftime("%Y-%m-%d")
                price_for_limit = vm_data["LAST_PRICE"] if vm_data.get("LAST_PRICE") is not None else vm_data["TODAY_PRICE"]
                limit_sum = (0.05 * price_for_limit * quantity * usd_rub) + (max(0, position_vm))
                vm_report = {
                    "TRADE_NAME": vm_data["TRADE_NAME"],
                    "SECID": vm_data["SECID"],
                    "TRADEDATE": vm_data["TRADEDATE"],
                    "LAST_SETTLE_PRICE": vm_data["LAST_SETTLE_PRICE"],
                    "TODAY_PRICE": vm_data["TODAY_PRICE"],
                    "LAST_PRICE": vm_data.get("LAST_PRICE"),
                    "QUOTE_TIME": vm_data.get("QUOTE_TIME"),
                    "PRICE_DATE": price_date,
                    "MULTIPLIER": vm_data["MULTIPLIER"],
                    "VM": vm_data["VM"],
                    "VM_CLEARING": vm_data["VM_CLEARING"],
                    "QUANTITY": quantity,
                    "POSITION_VM": position_vm,
                    "USD_RUB": usd_rub_data["usd_rub"],
                    "USD_RUB_DATE": usd_rub_data["date"],
                    "LIMIT_SUM": limit_sum,
                    "VM_SOURCE": "ISS MOEX",
                    "VAR_SOURCE": "Yahoo Finance",
                    "result_D": [],
                    "VAR_CONFIDENCE_LEVEL": 0.95,
                    "VAR_T": None,
                    "VAR_Q": None,
                    "VAR_K_VALUES": [],
                    "VAR_RESULTS": {},
                }
                try:
                    var_payload = get_gold_var_payload()
                    vm_report.update(
                        {
                            "result_D": var_payload["result_D"],
                            "VAR_CONFIDENCE_LEVEL": var_payload["confidence_level"],
                            "VAR_T": var_payload["T"],
                            "VAR_Q": var_payload["Q"],
                            "VAR_K_VALUES": var_payload["K_values"],
                            "VAR_RESULTS": var_payload["VAR_results"],
                        }
                    )
                except Exception as exc:
                    vm_report["VAR_ERROR"] = str(exc)
                try:
                    vm_report["XAUUSD_NEWS"] = fetch_xauusd_tradingview_news()
                except Exception as exc:
                    vm_report["XAUUSD_NEWS"] = []
                    vm_report["XAUUSD_NEWS_ERROR"] = str(exc)
                st.session_state["vm_last_report"] = vm_report
            except Exception as exc:
                st.error(str(exc))

    vm_report = st.session_state.get("vm_last_report")
    if vm_report:
        st.markdown(f"**Инструмент:** {vm_report['TRADE_NAME']}")
        st.markdown(f"**SECID:** {vm_report['SECID']}")
        st.markdown(f"**Дата клиринга:** {vm_report['TRADEDATE']}")
        st.markdown(f"**Расчетная цена последнего клиринга:** {vm_report['LAST_SETTLE_PRICE']}")
        st.markdown(f"**Последняя цена:** {vm_report.get('LAST_PRICE') if vm_report.get('LAST_PRICE') is not None else vm_report['TODAY_PRICE']}")
        st.markdown(f"**Дата последней цены:** {vm_report.get('PRICE_DATE', 'н/д')}")
        st.markdown(f"**Время котировки:** {vm_report.get('QUOTE_TIME') or 'н/д'}")
        st.markdown(f"**Multiplier:** {vm_report['MULTIPLIER']}")
        st.markdown(f"**Вариационная маржа (по последней цене):** {vm_report['VM']:.2f}")
        st.markdown(f"**VM клиринговая (SETTLEPRICEDAY - PREVSETTLEPRICE):** {vm_report.get('VM_CLEARING', vm_report['VM']):.2f}")
        st.markdown(f"**Маржа позиции (VM × Кол-во):** {safe_format_int_with_sep(vm_report['POSITION_VM'])}")
        st.markdown(f"**Сумма ограничения:** {safe_format_int_with_sep(vm_report['LIMIT_SUM'])}")
        st.caption(f"USD/RUB: {vm_report['USD_RUB']} на {vm_report['USD_RUB_DATE']}")

        st.markdown("#### Value at Risk (VaR)")
        st.caption(
            f"VM source: {vm_report.get('VM_SOURCE', 'ISS MOEX')}; "
            f"VaR/charts source: {vm_report.get('VAR_SOURCE', 'Yahoo Finance')}"
        )
        var_results = vm_report.get("VAR_RESULTS", {})
        if var_results:
            st.caption(
                f"Доверительный уровень: {vm_report.get('VAR_CONFIDENCE_LEVEL', 0.95):.2f}; "
                f"T = {vm_report.get('VAR_T', 0):.4f}; "
                f"Q = {vm_report.get('VAR_Q', 0):.6f}"
            )
            var_df = build_var_table({"VAR_results": var_results})
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        elif vm_report.get("VAR_ERROR"):
            st.info(f"VaR по данным Yahoo Finance недоступен: {vm_report['VAR_ERROR']}")
        else:
            st.info("Недостаточно дневной истории золота для расчёта VaR.")

        vm_df = pd.DataFrame([vm_report])
        vm_xlsx = ss.dataframe_to_excel_bytes(vm_df, sheet_name="vm_report")
        vm_pdf = build_vm_pdf_report(vm_report)
        st.download_button(
            label="💾 Скачать VM (Excel)",
            data=vm_xlsx,
            file_name="vm_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="vm_report_xlsx_dl",
        )
        st.download_button(
            label="💾 Скачать VM (PDF)",
            data=vm_pdf,
            file_name="vm_report.pdf",
            mime="application/pdf",
            key="vm_report_pdf_dl",
        )
        st.download_button(
            label="💾 Скачать VM (CSV)",
            data=vm_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="vm_report.csv",
            mime="text/csv",
            key="vm_report_csv_dl",
        )

        st.markdown("#### XAUUSD новости (TradingView)")
        news_items = vm_report.get("XAUUSD_NEWS", [])
        if news_items:
            st.caption("Период: со вчерашнего дня до последней доступной новости.")
            for item in news_items:
                st.markdown(
                    f"- **{item.get('published_at', '')}** — [{item.get('title', 'Без заголовка')}]({item.get('url', '')})"
                )
        elif vm_report.get("XAUUSD_NEWS_ERROR"):
            st.info(f"Новости XAUUSD временно недоступны: {vm_report['XAUUSD_NEWS_ERROR']}")
        else:
            st.info("За период со вчерашнего дня новости XAUUSD не найдены.")

        var_table_for_mail = build_var_table({"VAR_results": vm_report.get("VAR_RESULTS", {})})
        if not var_table_for_mail.empty:
            var_table_text = var_table_for_mail.to_string(index=False)
        elif vm_report.get("VAR_ERROR"):
            var_table_text = f"VaR по данным Yahoo Finance недоступен: {vm_report['VAR_ERROR']}"
        else:
            var_table_text = "Недостаточно дневной истории золота для расчёта VaR."
        mail_news_lines = [
            f"- {n.get('published_at', '')}: {n.get('title', '')}"
            for n in vm_report.get("XAUUSD_NEWS", [])[:10]
        ]
        mail_news_text = "\n".join(mail_news_lines) if mail_news_lines else "Нет доступных новостей за период."

        vm_mail_body = (
            "Коллеги, добрый день!\n\n"
            "Направляю отчёт по вариационной марже (VM).\n\n"
            f"Инструмент: {vm_report['TRADE_NAME']} ({vm_report['SECID']})\n"
            f"Дата клиринга: {vm_report['TRADEDATE']}\n"
            f"Кол-во: {vm_report['QUANTITY']}\n"
            f"VM (по последней цене): {vm_report['VM']:.2f}\n"
            f"VM клиринговая: {vm_report.get('VM_CLEARING', vm_report['VM']):.2f}\n"
            f"Маржа позиции: {safe_format_int_with_sep(vm_report['POSITION_VM'])}\n"
            f"Сумма ограничения: {safe_format_int_with_sep(vm_report['LIMIT_SUM'])}\n"
            f"USD/RUB: {vm_report['USD_RUB']} на {vm_report['USD_RUB_DATE']}\n\n"
            "Value at Risk (VaR):\n"
            f"{var_table_text}\n\n"
            "XAUUSD новости (TradingView, со вчерашнего дня):\n"
            f"{mail_news_text}\n\n"
            "Во вложении: Excel- и PDF-отчёты, а также график Gold Intraday (1M) - per gram.\n"
        )

        st.session_state["vm_report_default_body"] = vm_mail_body

        intraday_chart_attachment = None
        vm_pdf_attachment = ("vm_report.pdf", vm_pdf, "application", "pdf")
        try:
            render_gold_charts()
            intraday_chart_attachment = get_intraday_chart_attachment()
        except Exception as exc:
            st.warning(f"Не удалось построить графики по золоту: {exc}")

        render_email_compose_section(
            "VM отчёт",
            "vm_report",
            "vm_report.xlsx",
            vm_xlsx,
            extra_attachments=[att for att in [vm_pdf_attachment, intraday_chart_attachment] if att],
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
        st.markdown("#### Пакетный режим")
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
            "Q max (% от free-float капитализации, ось X)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            key="share_q_max",
            disabled=use_q_from_list,
        )
        use_log = st.checkbox("Логарифмическое приближение", value=True, key="share_q_log")
        q_mode = "log" if use_log else "linear"
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            share_calculate_clicked = st.button("Рассчитать Sell_stres (Share)", key="share_calculate")
        with action_col2:
            share_calculate_all_clicked = st.button(
                "Выгрузить модель по всем акциям",
                key="share_calculate_all",
                use_container_width=True,
            )

        if share_calculate_clicked or share_calculate_all_clicked:
            entries = []
            unresolved_identifiers = []
            if share_calculate_all_clicked:
                ranking_all_df = fetch_index_membership_by_isin(ALL_STOCK_INDEX_CODES)
                ranking_all_df = ranking_all_df.reindex(columns=["ISIN"], fill_value="")
                all_isins = sorted(
                    {
                        str(isin).strip().upper()
                        for isin in ranking_all_df["ISIN"].tolist()
                        if str(isin).strip()
                    }
                )
                prep_progress = st.progress(0.0)
                entries = []
                for idx, isin in enumerate(all_isins, start=1):
                    entries.append({"ISIN": isin, "Q_MAX": float(q_max)})
                    prep_progress.progress(idx / len(all_isins) if all_isins else 1.0)
                st.info(f"Подготовлено бумаг для полного расчёта: {len(entries)}")
            else:
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
                        entries.append({"ISIN": resolved_isin, "Q_MAX": float(q_val)})
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
                            entries.append({"ISIN": resolved_isin, "Q_MAX": float(q_max)})

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

                show_tables = len(entries) == 1 and not use_q_from_list and not share_calculate_all_clicked
                st.session_state["sell_stres_share_show_tables"] = show_tables
                st.session_state["sell_stres_share_table_results"] = results if show_tables else {}

                download_payload = {}
                if results:
                    combined_delta_df = pd.concat(
                        [df_delta.assign(ISIN=isin) for isin, df_delta in results.items()],
                        ignore_index=True,
                    )[["ISIN", "Q", "Q_RUB", "DeltaP"]]
                    ranking_df = fetch_index_membership_by_isin(ALL_STOCK_INDEX_CODES)
                    ranking_df = ranking_df.reindex(
                        columns=["ISIN", "Ticker", "Indices", "RankScore"],
                        fill_value="",
                    )
                    combined_delta_df = combined_delta_df.merge(
                        ranking_df,
                        on="ISIN",
                        how="left",
                    )
                    combined_delta_df["Ticker"] = combined_delta_df["Ticker"].fillna("")
                    combined_delta_df["Indices"] = combined_delta_df["Indices"].fillna("")
                    combined_delta_df["RankScore"] = combined_delta_df["RankScore"].fillna(0).astype(int)
                    download_payload["delta_csv"] = combined_delta_df.to_csv(index=False).encode("utf-8-sig")
                    download_payload["delta_xlsx"] = ss.dataframe_to_excel_bytes(
                        combined_delta_df, sheet_name="delta_p"
                    )
                    html_report = build_share_batch_html_report(
                        combined_delta_df=combined_delta_df[["ISIN", "Q", "Q_RUB", "DeltaP"]],
                        meta_df=pd.DataFrame(
                            meta_rows,
                            columns=[
                                "ISIN",
                                "SECID",
                                "T",
                                "Sigma",
                                "MDTV",
                                "Close",
                                "FreeFloat",
                                "IssueSize",
                                "FFShares",
                                "FFMcapRUB",
                            ],
                        )
                        if meta_rows
                        else pd.DataFrame(),
                        ranking_df=ranking_df,
                    )
                    download_payload["html_report"] = html_report
                    st.download_button(
                        label="💾 Скачать общий ΔP Excel",
                        data=ss.dataframe_to_excel_bytes(combined_delta_df, sheet_name="delta_p"),
                        file_name="sell_stres_share_deltaP_all.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                if meta_rows:
                    meta_df = pd.DataFrame(
                        meta_rows,
                        columns=[
                            "ISIN",
                            "SECID",
                            "T",
                            "Sigma",
                            "MDTV",
                            "Close",
                            "FreeFloat",
                            "IssueSize",
                            "FFShares",
                            "FFMcapRUB",
                        ],
                    )
                    ranking_df = fetch_index_membership_by_isin(ALL_STOCK_INDEX_CODES)
                    ranking_df = ranking_df.reindex(
                        columns=["ISIN", "Ticker", "Indices", "RankScore"],
                        fill_value="",
                    )
                    if not ranking_df.empty:
                        meta_df = meta_df.merge(ranking_df, on="ISIN", how="left")
                        meta_df["Ticker"] = meta_df["Ticker"].fillna("")
                        meta_df["Indices"] = meta_df["Indices"].fillna("")
                        meta_df["RankScore"] = meta_df["RankScore"].fillna(0).astype(int)
                        meta_df = meta_df.sort_values(["RankScore", "ISIN"], ascending=[False, True]).reset_index(drop=True)
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
            if "html_report" in share_downloads:
                st.download_button(
                    label="💾 Скачать веб-отчёт HTML (Share batch)",
                    data=share_downloads["html_report"],
                    file_name="sell_stres_share_batch_report.html",
                    mime="text/html",
                    key="share_html_report_dl",
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

@st.cache_data(ttl=900)
def load_moex_news(limit: int):
    return get_news(limit=limit)


@st.cache_data(ttl=900)
def load_moex_news_by_date(date_value: str):
    return get_news_by_date(date_value)


@st.cache_data(ttl=900)
def load_moex_news_by_isin(isin: str, days: int):
    return get_news_by_isin(isin, days=days)


def render_news_items(news_items: list[dict], empty_message: str) -> None:
    if not news_items:
        st.info(empty_message)
        return

    for item in news_items:
        item_datetime = item.get("datetime")
        datetime_label = item_datetime.strftime("%Y-%m-%d %H:%M:%S") if item_datetime else f"{item.get('date', '')} {item.get('time', '')}".strip()
        event_type = item.get("event_type") or "general"
        emitter = item.get("emitter") or "—"
        isins = item.get("related_isins") or []
        isin = ", ".join(isins) if isins else (item.get("isin") or "—")
        with st.container(border=True):
            st.markdown(f"**{item.get('title', 'Без заголовка')}**")
            body = str(item.get("body") or "").strip()
            if body:
                st.markdown(body)
            meta_left, meta_right = st.columns([2, 3])
            with meta_left:
                st.caption(f"{datetime_label} · {item.get('source', 'MOEX')}")
            with meta_right:
                st.caption(f"ID: {item.get('id', 'n/a')} · event_type: {event_type}")
            detail_left, detail_right = st.columns(2)
            detail_left.caption(f"Эмитент: {emitter}")
            detail_right.caption(f"ISIN: {isin}")
            source_link = str(item.get("link") or "").strip()
            if source_link:
                st.link_button("Открыть источник", source_link)


# ---------------------------
# MOEX news view
# ---------------------------
if st.session_state["active_view"] == "moex_news":
    st.subheader("📰 Новости MOEX")
    st.markdown("Поиск событий MOEX ISS `/iss/sitenews.json`: для вкладки ISIN поиск связанных новостей идет по всем другим ISIN того же эмитента, найденным через внутренний `emitent/emitter id`, а в карточке новости показывается полный текст, подтянутый через ID новости.")

    latest_col, date_col, isin_col = st.tabs(["Последние", "По дате", "По ISIN"])

    with latest_col:
        latest_limit = st.number_input(
            "Количество новостей", min_value=1, max_value=500, value=20, step=10, key="moex_news_limit"
        )
        if st.button("Загрузить последние новости", key="moex_news_load_latest"):
            try:
                latest_news = load_moex_news(limit=int(latest_limit))
            except (NewsServiceError, requests.RequestException) as exc:
                st.error(f"Не удалось загрузить новости MOEX: {exc}")
            else:
                st.success(f"Получено новостей: {len(latest_news)}")
                render_news_items(latest_news, "Нет новостей для отображения.")

    with date_col:
        selected_news_date = st.date_input(
            "Дата публикации",
            value=datetime.now().date(),
            key="moex_news_date_filter",
        )
        if st.button("Найти новости по дате", key="moex_news_load_by_date"):
            try:
                date_news = load_moex_news_by_date(selected_news_date.strftime("%Y-%m-%d"))
            except (ValueError, NewsServiceError, requests.RequestException) as exc:
                st.error(f"Не удалось получить новости по дате: {exc}")
            else:
                st.success(f"Новостей за {selected_news_date:%Y-%m-%d}: {len(date_news)}")
                render_news_items(date_news, "За выбранную дату новости не найдены.")

    with isin_col:
        news_isin = st.text_input(
            "ISIN",
            value="RU000A1008P1",
            placeholder="Например, RU000A1008P1",
            key="moex_news_isin_input",
        )
        related_days = st.number_input(
            "Глубина related_news, дней", min_value=0, max_value=365, value=7, step=1, key="moex_news_days"
        )
        if st.button("Найти новости по ISIN", key="moex_news_load_by_isin"):
            if not news_isin.strip():
                st.error("Введите ISIN для поиска новостей.")
            else:
                try:
                    isin_result = load_moex_news_by_isin(news_isin.strip().upper(), int(related_days))
                except (ValueError, NewsServiceError, requests.RequestException) as exc:
                    st.error(f"Не удалось получить новости по ISIN: {exc}")
                else:
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Target news", len(isin_result.get("target_news", [])))
                    metric_col2.metric("Related news", len(isin_result.get("related_news", [])))
                    metric_col3.metric("Other issuer ISINs", len(isin_result.get("other_isins", [])))
                    emitter_name = isin_result.get("emitter") or "—"
                    st.caption(f"Эмитент: {emitter_name}")
                    other_isins = isin_result.get("other_isins", [])
                    if other_isins:
                        st.caption("Другие ISIN этого эмитента: " + ", ".join(other_isins))
                    st.markdown("#### Target news")
                    render_news_items(
                        isin_result.get("target_news", []),
                        f"Новости с явным упоминанием ISIN {isin_result.get('isin', news_isin)} не найдены.",
                    )
                    st.markdown("#### Related news")
                    render_news_items(
                        isin_result.get("related_news", []),
                        "Связанные новости эмитента за выбранный период не найдены.",
                    )

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
                    board_df = pd.DataFrame(columns=["input", "SECID", "board", "turnover"])
                    st.info("Нет данных по boards за указанный период.")

                report_export_df = report_df.copy()
                board_export_df = board_df.copy()
                report_csv = report_export_df.to_csv(index=False).encode("utf-8-sig")
                board_csv = board_export_df.to_csv(index=False).encode("utf-8-sig")

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    report_export_df.to_excel(writer, index=False, sheet_name="turnover_totals")
                    board_export_df.to_excel(writer, index=False, sheet_name="turnover_by_board")
                excel_buffer.seek(0)

                st.markdown("### Скачать отчеты")
                download_col_left, download_col_right = st.columns(2)
                with download_col_left:
                    st.download_button(
                        label="💾 Оборот по инструментам (CSV)",
                        data=report_csv,
                        file_name=f"moex_turnover_totals_{start_date_input}_{end_date_input}.csv",
                        mime="text/csv",
                        key="moex_turnover_totals_csv",
                    )
                    st.download_button(
                        label="💾 Оборот по boards (CSV)",
                        data=board_csv,
                        file_name=f"moex_turnover_boards_{start_date_input}_{end_date_input}.csv",
                        mime="text/csv",
                        key="moex_turnover_boards_csv",
                    )
                with download_col_right:
                    st.download_button(
                        label="💾 Полный отчет (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name=f"moex_turnover_{start_date_input}_{end_date_input}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="moex_turnover_excel",
                    )

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
    exclude_etf = st.checkbox(
        "Исключать ETF из расчета",
        value=False,
        key="market_statistics_exclude_etf",
        help="Фильтрует инструменты, где название содержит ETF (например, LQDT ETF).",
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
                if exclude_etf:
                    shortname_series = combined_df.get("SHORTNAME", pd.Series("", index=combined_df.index)).astype(str)
                    combined_df = combined_df[~shortname_series.str.contains(r"\bETF\b", case=False, na=False)]
                    if combined_df.empty:
                        st.error("После исключения ETF данные отсутствуют.")
                        st.stop()
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

            unique_identifiers = list(dict.fromkeys(raw_identifiers))
            total_identifiers = len(unique_identifiers)
            progress_bar = st.progress(0, text="Подготовка к загрузке оборотов...")

            with st.spinner("Загружаем обороты..."):
                for idx, identifier in enumerate(unique_identifiers, start=1):
                    progress_bar.progress(
                        int((idx - 1) / max(total_identifiers, 1) * 100),
                        text=f"Обрабатываем {idx}/{total_identifiers}: {identifier}",
                    )
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

                progress_bar.progress(100, text="Загрузка оборотов завершена")

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
                    on_click="ignore",
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
                        on_click="ignore",
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
