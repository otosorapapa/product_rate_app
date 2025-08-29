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
    params, warn1 = parse_hyochin(xls)

with st.spinner("『R6.12』製品データを解析中..."):
    df_products, warn2 = parse_products(xls, sheet_name="R6.12")

for w in (warn1 + warn2):
    st.warning(w)

st.session_state["va_params"] = params
st.session_state["df_products_raw"] = df_products
st.session_state["be_rate"] = float(params.get("break_even_rate") or 0.0)
st.session_state["req_rate"] = float(params.get("required_rate") or 0.0)

c1, c2, c3 = st.columns(3)
c1.metric("固定費（計）", f"{params.get('fixed_total'):,}" if params.get('fixed_total') else "-")
c2.metric("必要利益（計）", f"{params.get('required_profit_total'):,}" if params.get('required_profit_total') else "-")
c3.metric("年間標準稼働時間（分）", f"{params.get('annual_minutes'):,}" if params.get('annual_minutes') else "-")

st.divider()
st.subheader("製品データ（先頭20件プレビュー）")
st.dataframe(df_products.head(20), use_container_width=True)

st.success("保存しました。上部のナビから『ダッシュボード』へ進んでください。")
