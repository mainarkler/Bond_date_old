from __future__ import annotations

from datetime import datetime
from io import BytesIO
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd


def build_share_batch_xml_report(
    combined_delta_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> bytes:
    root = ET.Element("sellStressBatchReport", generatedAt=datetime.utcnow().isoformat(timespec="seconds"))
    summary = ET.SubElement(root, "summary")
    ET.SubElement(summary, "isinsTotal").text = str(combined_delta_df["ISIN"].nunique())
    ET.SubElement(summary, "rowsTotal").text = str(len(combined_delta_df))

    ranking_map = ranking_df.set_index("ISIN").to_dict("index") if not ranking_df.empty else {}
    meta_map = meta_df.set_index("ISIN").to_dict("index") if not meta_df.empty else {}

    for isin, df_isin in combined_delta_df.groupby("ISIN"):
        rank_item = ranking_map.get(isin, {})
        node = ET.SubElement(
            root,
            "security",
            isin=str(isin),
            indices=str(rank_item.get("Indices", "")),
            rankScore=str(rank_item.get("RankScore", 0)),
        )

        meta_node = ET.SubElement(node, "meta")
        for key, value in meta_map.get(isin, {}).items():
            ET.SubElement(meta_node, key).text = "" if pd.isna(value) else str(value)

        points = ET.SubElement(node, "curve")
        for row in df_isin.sort_values("Q").itertuples(index=False):
            ET.SubElement(
                points,
                "point",
                q=str(int(row.Q)),
                deltaP=str(float(row.DeltaP)),
            )

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_share_batch_png_chart(combined_delta_df: pd.DataFrame, ranking_df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 7))
    rank_map = ranking_df.set_index("ISIN").to_dict("index") if not ranking_df.empty else {}

    for isin, df_isin in combined_delta_df.groupby("ISIN"):
        rank_item = rank_map.get(isin, {})
        label = f"{isin} [{rank_item.get('Indices', '-')} ]"
        ax.plot(df_isin["Q"], df_isin["DeltaP"], linewidth=1.5, alpha=0.9, label=label)

    ax.set_title("Sell_stres Share: ΔP by Q")
    ax.set_xlabel("Q (volume sold)")
    ax.set_ylabel("ΔP")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 12:
        handles, labels = handles[:12], labels[:12]
    if labels:
        ax.legend(handles, labels, fontsize=8, loc="best")

    output = BytesIO()
    fig.tight_layout()
    fig.savefig(output, format="png", dpi=160)
    plt.close(fig)
    output.seek(0)
    return output.getvalue()
