import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils import compute_results

st.title("② ダッシュボード")

if "df_products_raw" not in st.session_state or st.session_state["df_products_raw"] is None or len(st.session_state["df_products_raw"]) == 0:
    st.info("先に『① データ入力 & 取り込み』でデータを準備してください。")
    st.stop()

df_products_raw = st.session_state["df_products_raw"]
be_rate = st.session_state.get("be_rate", 0.0)
req_rate = st.session_state.get("req_rate", 0.0)

# Controls
with st.expander("基準賃率の調整（What-if）", expanded=False):
    colA, colB, colC = st.columns([1,1,1.5])
    be_rate = colA.number_input("損益分岐賃率（円/分）", value=float(be_rate), step=0.001, format="%.6f")
    req_rate = colB.number_input("必要賃率（円/分）", value=float(req_rate), step=0.001, format="%.6f")
    topn = int(colC.slider("未達SKUの上位件数（テーブル/パレート）", min_value=5, max_value=50, value=20, step=1))

df = compute_results(df_products_raw, be_rate, req_rate)

# Global filters
fcol1, fcol2, fcol3 = st.columns([1,1,2])
classes = df["rate_class"].dropna().unique().tolist()
selected_classes = fcol1.multiselect("達成分類で絞り込み", classes, default=classes)
search = fcol2.text_input("製品名 検索（部分一致）", "")
mpu_min, mpu_max = fcol3.slider("分/個（製造リードタイム）の範囲", float(np.nan_to_num(df["minutes_per_unit"].min(), nan=0.0)), float(np.nan_to_num(df["minutes_per_unit"].max(), nan=10.0)), value=(0.0, float(np.nan_to_num(df["minutes_per_unit"].max(), nan=10.0))))

mask = df["rate_class"].isin(selected_classes)
if search:
    mask &= df["product_name"].astype(str).str.contains(search, na=False)
mask &= df["minutes_per_unit"].fillna(0.0).between(mpu_min, mpu_max)
df_view = df[mask].copy()

# KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("必要賃率 (円/分)", f"{req_rate:,.3f}")
col2.metric("損益分岐賃率 (円/分)", f"{be_rate:,.3f}")
ach_rate = (df_view["meets_required_rate"].mean()*100.0) if len(df_view)>0 else 0.0
col3.metric("必要賃率達成SKU比率", f"{ach_rate:,.1f}%")
avg_vapm = df_view["va_per_min"].replace([np.inf,-np.inf], np.nan).dropna().mean() if "va_per_min" in df_view else 0.0
col4.metric("平均 付加価値/分", f"{avg_vapm:,.1f}")

tabs = st.tabs(["全体分布（散布図）", "達成状況（棒/円）", "未達SKU（パレート）", "SKUテーブル"])

with tabs[0]:
    st.caption("横軸=分/個（製造リードタイム）, 縦軸=付加価値/分。赤線=必要賃率。")
    base = alt.Chart(df_view).mark_circle().encode(
        x=alt.X("minutes_per_unit:Q", title="分/個"),
        y=alt.Y("va_per_min:Q", title="付加価値/分"),
        tooltip=["product_name:N","minutes_per_unit:Q","va_per_min:Q","rate_class:N"]
    ).properties(height=420)
    rule = alt.Chart(pd.DataFrame({"y":[req_rate]})).mark_rule(color="red").encode(y="y:Q")
    color = base.encode(color=alt.Color("rate_class:N", legend=alt.Legend(title="分類")))
    st.altair_chart(color + rule, use_container_width=True)

with tabs[1]:
    c1, c2 = st.columns([1.2,1])
    class_counts = df_view["rate_class"].value_counts().reset_index()
    class_counts.columns = ["rate_class", "count"]
    bar = alt.Chart(class_counts).mark_bar().encode(
        x=alt.X("rate_class:N", title="達成分類"),
        y=alt.Y("count:Q", title="件数"),
        tooltip=["rate_class","count"]
    ).properties(height=380)
    c1.altair_chart(bar, use_container_width=True)

    # Achievers vs Missed donut
    donut_df = pd.DataFrame({
        "group": ["達成", "未達"],
        "value": [ (df_view["meets_required_rate"].sum()), ( (~df_view["meets_required_rate"]).sum() ) ]
    })
    donut = alt.Chart(donut_df).mark_arc(innerRadius=80).encode(theta="value:Q", color="group:N", tooltip=["group","value"])
    c2.altair_chart(donut, use_container_width=True)

with tabs[2]:
    miss = df_view[df_view["meets_required_rate"] == False].copy()
    miss = miss.sort_values("rate_gap_vs_required").head(topn)
    st.caption("『必要賃率差』が小さい（またはマイナスが大）の順。右ほど改善余地が大。")
    if len(miss)==0:
        st.success("未達SKUはありません。")
    else:
        pareto = alt.Chart(miss).mark_bar().encode(
            x=alt.X("product_name:N", sort="-y", title="製品名"),
            y=alt.Y("rate_gap_vs_required:Q", title="必要賃率差（付加価値/分 - 必要賃率）"),
            tooltip=["product_name","rate_gap_vs_required"]
        ).properties(height=420)
        st.altair_chart(pareto, use_container_width=True)
        st.dataframe(miss[["product_no","product_name","minutes_per_unit","va_per_min","rate_gap_vs_required","price_gap_vs_required"]], use_container_width=True)

with tabs[3]:
    rename_map = {
        "product_no": "製品番号",
        "product_name": "製品名",
        "actual_unit_price": "実際売単価",
        "material_unit_cost": "材料原価",
        "minutes_per_unit": "分/個",
        "daily_qty": "日産数",
        "daily_total_minutes": "日産合計(分)",
        "gp_per_unit": "粗利/個",
        "daily_va": "付加価値(日産)",
        "va_per_min": "付加価値/分",
        "be_va_unit_price": "損益分岐付加価値単価",
        "req_va_unit_price": "必要付加価値単価",
        "required_selling_price": "必要販売単価",
        "price_gap_vs_required": "必要販売単価差額",
        "rate_gap_vs_required": "必要賃率差",
        "meets_required_rate": "必要賃率達成",
        "rate_class": "達成分類",
    }
    ordered_cols = [
        "製品番号","製品名","実際売単価","必要販売単価","必要販売単価差額","材料原価","粗利/個",
        "分/個","日産数","日産合計(分)","付加価値(日産)","付加価値/分",
        "損益分岐付加価値単価","必要付加価値単価","必要賃率差","必要賃率達成","達成分類",
    ]
    df_table = df_view.rename(columns=rename_map)
    df_table = df_table[[c for c in ordered_cols if c in df_table.columns]]

    st.dataframe(df_table, use_container_width=True, height=520)
    csv = df_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("結果をCSVでダウンロード", data=csv, file_name="calc_results.csv", mime="text/csv")
