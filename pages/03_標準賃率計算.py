import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import streamlit as st
import numpy as np
import pandas as pd

from standard_rate_core import (
    DEFAULT_PARAMS,
    sanitize_params,
    compute_rates,
    build_reverse_index,
    plot_sensitivity,
    generate_pdf,
)

st.title("③ 標準賃率 計算/感度分析")
scenarios = st.session_state.setdefault("scenarios", {"Base": st.session_state.get("sr_params", DEFAULT_PARAMS)})
current = st.session_state.setdefault("current_scenario", "Base")
st.caption(f"適用中シナリオ: {current}")

params = scenarios.get(current, st.session_state.get("sr_params", DEFAULT_PARAMS)).copy()

st.sidebar.header("シナリオ")
names = list(scenarios.keys())
selected = st.sidebar.selectbox("シナリオ選択", names, index=names.index(current))
if selected != current:
    st.session_state["current_scenario"] = selected
    st.session_state["sr_params"] = scenarios[selected].copy()
    st.experimental_rerun()

new_name = st.sidebar.text_input("新規シナリオ名", "")
if st.sidebar.button("追加") and new_name:
    scenarios[new_name] = params.copy()
    st.session_state["current_scenario"] = new_name
    st.session_state["sr_params"] = params.copy()
    st.experimental_rerun()

if current != "Base" and st.sidebar.button("削除"):
    del scenarios[current]
    st.session_state["current_scenario"] = "Base"
    st.session_state["sr_params"] = scenarios["Base"].copy()
    st.experimental_rerun()

st.sidebar.header("入力")
placeholders = {}

st.sidebar.subheader("A) 必要固定費（円/年）")
params["labor_cost"] = st.sidebar.number_input("労務費", value=float(params["labor_cost"]), step=1.0, format="%.0f", min_value=0.0)
placeholders["labor_cost"] = st.sidebar.empty()
params["sga_cost"] = st.sidebar.number_input("販管費", value=float(params["sga_cost"]), step=1.0, format="%.0f", min_value=0.0)
placeholders["sga_cost"] = st.sidebar.empty()

st.sidebar.subheader("B) 必要利益（円/年）")
params["loan_repayment"] = st.sidebar.number_input("借入返済（年）", value=float(params["loan_repayment"]), step=1.0, format="%.0f", min_value=0.0)
placeholders["loan_repayment"] = st.sidebar.empty()
params["tax_payment"] = st.sidebar.number_input("納税・納付", value=float(params["tax_payment"]), step=1.0, format="%.0f", min_value=0.0)
placeholders["tax_payment"] = st.sidebar.empty()
params["future_business"] = st.sidebar.number_input("未来事業費", value=float(params["future_business"]), step=1.0, format="%.0f", min_value=0.0)
placeholders["future_business"] = st.sidebar.empty()

st.sidebar.subheader("C) 工数前提")
params["fulltime_workers"] = st.sidebar.number_input("正社員：人数", value=float(params["fulltime_workers"]), step=1.0, format="%.2f", min_value=0.0)
placeholders["fulltime_workers"] = st.sidebar.empty()
st.sidebar.caption("労働係数=1.00")
params["part1_workers"] = st.sidebar.number_input("準社員①：人数", value=float(params["part1_workers"]), step=1.0, format="%.2f", min_value=0.0)
placeholders["part1_workers"] = st.sidebar.empty()
st.sidebar.caption("準社員① 労働係数=0.75")
params["part2_workers"] = st.sidebar.number_input("準社員②：人数", value=float(params["part2_workers"]), step=1.0, format="%.2f", min_value=0.0)
placeholders["part2_workers"] = st.sidebar.empty()
params["part2_coefficient"] = st.sidebar.slider("準社員②：労働係数", value=float(params["part2_coefficient"]), min_value=0.0, max_value=1.0, step=0.01)
placeholders["part2_coefficient"] = st.sidebar.empty()

params["working_days"] = st.sidebar.number_input("年間稼働日数（日）", value=float(params["working_days"]), step=1.0, format="%.0f", min_value=1.0)
placeholders["working_days"] = st.sidebar.empty()
params["daily_hours"] = st.sidebar.number_input("1日当り稼働時間（時間）", value=float(params["daily_hours"]), step=0.1, format="%.2f", min_value=0.1)
placeholders["daily_hours"] = st.sidebar.empty()
params["operation_rate"] = st.sidebar.slider("1日当り操業度", value=float(params["operation_rate"]), min_value=0.5, max_value=1.0, step=0.01)
placeholders["operation_rate"] = st.sidebar.empty()

params, warn_list = sanitize_params(params)
for w in warn_list:
    st.sidebar.warning(w)
st.session_state["sr_params"] = params
scenarios[current] = params
st.session_state["scenarios"] = scenarios

nodes, results = compute_rates(params)
reverse_index = build_reverse_index(nodes)
for k, ph in placeholders.items():
    affected = ", ".join(reverse_index.get(k, []))
    if affected:
        ph.caption(f"この入力が影響する指標: {affected}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("損益分岐賃率（円/分）", f"{results['break_even_rate']:.3f}")
c2.metric("必要賃率（円/分）", f"{results['required_rate']:.3f}")
c3.metric("年間標準稼働時間（分）", f"{results['annual_minutes']:.0f}")
c4.metric("正味直接工員数合計", f"{results['net_workers']:.2f}")

st.subheader("ブレークダウン")
cat_map = {
    "fixed_total": "固定費",
    "required_profit_total": "必要利益",
    "net_workers": "工数前提",
    "minutes_per_day": "工数前提",
    "standard_daily_minutes": "工数前提",
    "annual_minutes": "工数前提",
    "break_even_rate": "賃率",
    "required_rate": "賃率",
    "daily_be_va": "付加価値",
    "daily_req_va": "付加価値",
}
df_break = pd.DataFrame(
    [
        (
            cat_map.get(n["key"], ""),
            n["label"],
            n["value"],
            n.get("unit", ""),
            n["formula"],
            ", ".join(n["depends_on"]),
        )
        for n in nodes.values()
    ],
    columns=["区分", "項目", "値", "単位", "式", "依存要素"],
)
st.dataframe(df_break, use_container_width=True)

st.subheader("感度分析")
fig = plot_sensitivity(params)
st.pyplot(fig)

df_csv = pd.DataFrame(list(nodes.values()))
df_csv["depends_on"] = df_csv["depends_on"].apply(lambda x: ",".join(x))
csv = df_csv.to_csv(index=False, encoding="utf-8-sig")
st.download_button("CSVエクスポート", data=csv, file_name=f"standard_rate__{current}.csv", mime="text/csv")

pdf_bytes = generate_pdf(nodes, fig)
st.download_button("PDFエクスポート", data=pdf_bytes, file_name=f"standard_rate_summary__{current}.pdf", mime="application/pdf")
