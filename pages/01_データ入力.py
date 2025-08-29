import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    read_excel_safely,
    parse_hyochin,
    parse_products,
    compute_results,
    PRODUCT_SCHEMA,
    PROCESS_COLS,
)
from standard_rate_core import compute_rates

st.title("① データ入力 & 取り込み")

st.markdown("""
### このページで行うこと
- **PDF「データ入力シート」準拠**で、全項目を取り込み・正規化します（工程列は分換算）。
- 取り込んだ諸元から、**分/個・日産合計(分)・粗利/個・付加価値/分・必要/損益分岐単価**を**自動再計算**します。
- **シミュレーション（係数適用）**で、価格・材料費・外注費・工程時間・日産数を一括調整し、**差分を即時可視化**します。
- 以降のページ（**②ダッシュボード／③感度分析**）でも**同じ賃率・KPI**が連動します。
""")

# ---------------- Step1: 取込 -----------------
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

_, rate_res = compute_rates(sr_params)
be_rate = rate_res["break_even_rate"]
req_rate = rate_res["required_rate"]

df_products = compute_results(df_products, be_rate, req_rate)

st.session_state["sr_params"] = sr_params
st.session_state["df_products_raw"] = df_products
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {"Base": sr_params.copy()}
    st.session_state["current_scenario"] = "Base"
else:
    st.session_state["scenarios"].setdefault("Base", sr_params.copy())
    st.session_state.setdefault("current_scenario", "Base")

st.caption(f"適用中シナリオ: {st.session_state['current_scenario']}")

st.subheader("1) 取込サマリ")
st.write(f"読み込み件数: {len(df_products)}")

# ---------------- Step2: 項目マッピング -----------------
st.subheader("2) 項目マッピング")
show_proc = st.checkbox("工程列も表示", value=False)
mapping = []
for key, meta in PRODUCT_SCHEMA.items():
    mapping.append({
        "内部キー": key,
        "表示名": meta["label"],
        "単位": meta.get("unit", ""),
        "存在": key in df_products.columns,
    })
if show_proc:
    for p in PROCESS_COLS:
        mapping.append({"内部キー": p, "表示名": p, "単位": "分", "存在": p in df_products.columns})
map_df = pd.DataFrame(mapping)
st.dataframe(map_df, use_container_width=True, height=300)
missing = [m["表示名"] for m in mapping if not m["存在"]]
if missing:
    st.warning("未対応列: " + ", ".join(missing))

# ---------------- Sidebar: Simulation controls -----------------
st.sidebar.header("シミュレーション")
price_pct = st.sidebar.number_input("実際売単価(±%)", value=0.0, step=1.0)
mat_pct = st.sidebar.number_input("原価(±%)", value=0.0, step=1.0)
sub_pct = st.sidebar.number_input("外注費(±%)", value=0.0, step=1.0)
proc_pct = st.sidebar.number_input("工程時間(±%)", value=0.0, step=1.0)
qty_pct = st.sidebar.number_input("日産数(±%)", value=0.0, step=1.0)

st.sidebar.subheader("フィルタ")
name_filter = st.sidebar.text_input("製品名（部分一致）", "")
mpu_min = float(np.nan_to_num(df_products["minutes_per_unit"].min(), nan=0.0))
mpu_max = float(np.nan_to_num(df_products["minutes_per_unit"].max(), nan=0.0))
mpu_range = st.sidebar.slider("分/個の範囲", mpu_min, max(mpu_min, mpu_max), (mpu_min, max(mpu_min, mpu_max)))

if st.sidebar.button("プレビュー更新"):
    df_new = st.session_state["df_products_raw"].copy()
    mask = pd.Series(True, index=df_new.index)
    if name_filter:
        mask &= df_new["product_name"].astype(str).str.contains(name_filter, na=False)
    mask &= df_new["minutes_per_unit"].fillna(0.0).between(mpu_range[0], mpu_range[1])
    df_new.loc[mask, "actual_unit_price"] *= (1 + price_pct / 100.0)
    df_new.loc[mask, "material_unit_cost"] *= (1 + mat_pct / 100.0)
    df_new.loc[mask, "subcontract_cost"] *= (1 + sub_pct / 100.0)
    df_new.loc[mask, PROCESS_COLS] *= (1 + proc_pct / 100.0)
    df_new.loc[mask, "daily_qty"] *= (1 + qty_pct / 100.0)
    df_new["minutes_per_unit"] = df_new[PROCESS_COLS].sum(axis=1, skipna=True)
    _, rate_res = compute_rates(st.session_state["sr_params"])
    df_new = compute_results(df_new, rate_res["break_even_rate"], rate_res["required_rate"])
    st.session_state["df_products_sim"] = df_new
elif st.sidebar.button("Baseに戻す"):
    st.session_state.pop("df_products_sim", None)
    st.session_state["current_scenario"] = "Base"

scenario_name = st.sidebar.text_input("シナリオ保存名", "")
if st.sidebar.button("シナリオへ保存") and scenario_name:
    df_save = st.session_state.get("df_products_sim", st.session_state["df_products_raw"]).copy()
    st.session_state.setdefault("product_scenarios", {})[scenario_name] = df_save
    st.session_state.setdefault("scenarios", {})[scenario_name] = st.session_state.get("sr_params", {}).copy()
    st.session_state["df_products_sim"] = df_save
    st.session_state["current_scenario"] = scenario_name

# ---------------- Step4: プレビュー/検証 -----------------
st.subheader("3) シミュレーション")
st.caption("サイドバーで係数を調整し、プレビューを更新してください。")

df_display = st.session_state.get("df_products_sim", st.session_state["df_products_raw"])
_, rate_res = compute_rates(st.session_state["sr_params"])
req_rate = rate_res["required_rate"]
be_rate = rate_res["break_even_rate"]
avg_vapm = df_display["va_per_min"].replace([np.inf, -np.inf], np.nan).dropna().mean()
median_mpu = df_display["minutes_per_unit"].replace([np.inf, -np.inf], np.nan).dropna().median()
col1, col2, col3, col4 = st.columns(4)
col1.metric("必要賃率 (円/分)", f"{req_rate:.3f}")
col2.metric("損益分岐賃率 (円/分)", f"{be_rate:.3f}")
col3.metric("平均VA/分", f"{avg_vapm:.1f}")
col4.metric("工程合計(中央値)", f"{median_mpu:.1f}")

st.subheader("4) プレビュー/検証")
base = st.session_state["df_products_raw"].head(50)
preview = df_display.head(50)

def highlight_changes(data: pd.DataFrame) -> np.ndarray:
    b = base.reindex(data.index)[data.columns]
    diff = data.ne(b)
    return np.where(diff, "background-color:#ffe0b2", "")

st.dataframe(preview.style.apply(highlight_changes, axis=None), use_container_width=True)

csv = df_display.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSVダウンロード", data=csv, file_name="products_sim.csv", mime="text/csv")

try:
    st.page_link("pages/02_ダッシュボード.py", label="② ダッシュボードへ")
except Exception:
    st.markdown("[② ダッシュボードへ](./02_ダッシュボード)")
