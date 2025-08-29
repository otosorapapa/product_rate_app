import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from utils import read_excel_safely, parse_hyochin, parse_products

st.title("① データ入力 & 取り込み")

default_path = "data/sample.xlsx"
file = st.file_uploader("Excelをアップロード（未指定ならサンプルを使用）", type=["xlsx"])

if file is None:
    st.info("サンプルデータを使用します。")
    xls = read_excel_safely(default_path)
else:
    xls = read_excel_safely(file)

if xls is None:
    st.error("Excel 読込に失敗しました。ファイル形式・シート名をご確認ください。")
    st.stop()

with st.spinner("『標賃』を解析中..."):
    calc_params, sr_params, warn1 = parse_hyochin(xls)

with st.spinner("『R6.12』製品データを解析中..."):
    df_products, warn2 = parse_products(xls, sheet_name="R6.12")

for w in (warn1 + warn2):
    st.warning(w)

st.session_state["sr_params"] = sr_params
st.session_state["df_products_raw"] = df_products
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {"Base": sr_params.copy()}
    st.session_state["current_scenario"] = "Base"
else:
    st.session_state["scenarios"]["Base"] = sr_params.copy()
if "current_scenario" not in st.session_state:
    st.session_state["current_scenario"] = "Base"

st.caption(f"適用中シナリオ: {st.session_state['current_scenario']}")

c1, c2, c3 = st.columns(3)
c1.metric("固定費計 (円/年)", f"{calc_params.get('fixed_total', 0):,.0f}")
c2.metric("必要利益計 (円/年)", f"{calc_params.get('required_profit_total', 0):,.0f}")
c3.metric("年間標準稼働分 (分/年)", f"{calc_params.get('annual_minutes', 0):,.0f}")

st.divider()
st.subheader("製品データ（先頭20件プレビュー）")
st.dataframe(df_products.head(20), use_container_width=True)

st.success("保存しました。上部のナビから『ダッシュボード』へ進んでください。")
