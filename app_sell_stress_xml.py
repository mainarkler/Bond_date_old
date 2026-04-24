from __future__ import annotations

from datetime import date
from pathlib import Path

import altair as alt
import streamlit as st

from sell_stress_ui.data import filter_assets, load_asset_universe
from sell_stress_ui.form_config import load_form_config
from sell_stress_ui.service import SellStressRequest, calculate_price_impact

XML_FORM_PATH = Path(__file__).resolve().parent / "sell_stress_ui" / "schemas" / "sell_stress_form.xml"


@st.cache_data(show_spinner=False)
def get_form_config():
    return load_form_config(XML_FORM_PATH)


@st.cache_data(show_spinner=False)
def get_asset_universe():
    return load_asset_universe()


def run() -> None:
    cfg = get_form_config()
    st.set_page_config(page_title=cfg.title, page_icon="📉", layout="wide")
    st.title(cfg.title)
    st.caption(cfg.description)

    universe = get_asset_universe()

    col1, col2 = st.columns(2)
    with col1:
        index_filter = st.selectbox(
            cfg.fields["index_filter"].label,
            options=["ALL", "IMOEX", "RTS"],
            index=0,
        )
    with col2:
        stock_filter = st.text_input(cfg.fields["asset_filter"].label, placeholder="SBER")

    filtered = filter_assets(universe, index_filter=index_filter, stock_filter=stock_filter)
    if filtered.empty:
        st.warning("No assets found with current filters.")
        st.stop()

    asset_options = {
        f"{row.symbol} — {row.name} ({row.isin})": row for row in filtered.itertuples(index=False)
    }
    selected_label = st.selectbox(cfg.fields["asset_selector"].label, options=list(asset_options.keys()))
    selected = asset_options[selected_label]

    volume = st.number_input(
        f'{cfg.fields["volume"].label} (% от free-float капитализации)',
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
    )
    c_value = st.number_input(cfg.fields["c_value"].label, min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    date_from = st.date_input(cfg.fields["date_from"].label, value=date(2024, 1, 1))
    q_mode = st.selectbox(cfg.fields["q_mode"].label, options=["log", "linear"], index=0)

    if st.button("Calculate sell stress", type="primary"):
        if volume <= 0:
            st.error("Volume must be positive.")
            st.stop()

        req = SellStressRequest(
            isin=selected.isin,
            secid=selected.secid,
            volume=float(volume),
            c_value=float(c_value),
            date_from=date_from.strftime("%Y-%m-%d"),
            q_mode=q_mode,
        )

        with st.spinner("Running sell_stress..."):
            result_df, meta = calculate_price_impact(req)

        st.subheader("Results")
        st.json(meta)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        chart = (
            alt.Chart(result_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Q:Q", title="% от free-float капитализации"),
                y=alt.Y("PriceAfterSell:Q", title="Price after sell (baseline=100)"),
                tooltip=["Q", "DeltaP", "DrawdownPct", "PriceAfterSell"],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

        st.download_button(
            "Export CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"sell_stress_{selected.symbol}_{index_filter.lower()}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    run()
