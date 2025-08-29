
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any

st.set_page_config(page_title="製品賃率計算アプリ", layout="wide")

# ============== Utilities ==============
def _clean(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\n", "").strip()

def read_excel_safely(path_or_bytes):
    try:
        xls = pd.ExcelFile(path_or_bytes, engine="openpyxl")
        return xls
    except Exception as e:
        st.error(f"Excel読込エラー: {e}")
        return None

def find_header_row(df: pd.DataFrame, keyword: str) -> int:
    for i in range(len(df)):
        if (df.iloc[i] == keyword).any():
            return i
    return -1

def build_columns_from_two_rows(header_row: pd.Series, unit_row: pd.Series) -> List[str]:
    cols = []
    for h, u in zip(header_row, unit_row):
        h2 = _clean(h); u2 = _clean(u)
        if not h2:
            cols.append("")
        elif u2:
            cols.append(f"{h2} ({u2})")
        else:
            cols.append(h2)
    return cols

def series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series or NaN Series aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def classify_rate_gap(gap: float) -> str:
    """分類用に賃率差を評価する"""
    if pd.isna(gap):
        return "不明"
    if gap >= 1.0:
        return "余裕あり"
    if gap >= 0:
        return "達成"
    return "未達"

# ============== Parsing 標賃 ==============
def parse_hyochin(xls: pd.ExcelFile) -> Dict[str, Any]:
    try:
        df = pd.read_excel(xls, sheet_name="標賃", header=None)
    except Exception as e:
        st.warning(f"シート『標賃』が読めません: {e}")
        return {}

    def get_by_label(label:str):
        rows = df[(df.iloc[:,1].astype(str).str.contains(label, na=False))]
        if rows.empty:
            return None
        row = rows.iloc[0]
        val = None
        for x in row[::-1]:
            try:
                xv = float(x)
                val = xv
                break
            except:
                continue
        return val

    labor = get_by_label("労務費")
    sgna = get_by_label("販管費")
    fixed_total = get_by_label("計")  # ➀必要固定費の計（最初の計を想定）

    required_profit_total = None
    idx_required_block = df.index[df.iloc[:,1].astype(str).str.contains("②必要利益", na=False)]
    if len(idx_required_block)>0:
        start = idx_required_block[0]
        for i in range(start, min(start+10, len(df))):
            if "計" in str(df.iloc[i,1]):
                try:
                    required_profit_total = float(df.iloc[i,3])
                    break
                except:
                    pass

    annual_minutes = None
    rows = df[(df.iloc[:,1].astype(str).str.contains("年間標準稼働時間（分）", na=False))]
    if not rows.empty:
        try:
            annual_minutes = float(rows.iloc[0,3])
        except:
            annual_minutes = None

    be_rate = None; req_rate = None
    rows_be = df[(df.iloc[:,1].astype(str).str.contains("損益分岐賃率", na=False))]
    if not rows_be.empty:
        try:
            be_rate = float(rows_be.iloc[0,3])
        except:
            pass
    rows_req = df[(df.iloc[:,1].astype(str).str.contains("必要賃率", na=False))]
    if not rows_req.empty:
        try:
            req_rate = float(rows_req.iloc[0,3])
        except:
            pass

    if (be_rate is None or req_rate is None) and fixed_total and annual_minutes:
        try:
            if be_rate is None:
                be_rate = fixed_total / annual_minutes
            if req_rate is None and required_profit_total:
                req_rate = (fixed_total + required_profit_total) / annual_minutes
        except:
            pass

    return dict(
        labor_cost=labor,
        sgna=sgna,
        fixed_total=fixed_total,
        required_profit_total=required_profit_total,
        annual_minutes=annual_minutes,
        break_even_rate=be_rate,
        required_rate=req_rate,
    )

# ============== Parsing R6.12（製品） ==============
def parse_products(xls: pd.ExcelFile, sheet_name: str="R6.12") -> pd.DataFrame:
    try:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
        st.warning(f"シート『{sheet_name}』が読めません: {e}")
        return pd.DataFrame()

    hdr_row = find_header_row(raw, "製品№")
    if hdr_row < 0:
        st.warning("『製品№』行が見つかりません。")
        return pd.DataFrame()

    header_row = raw.iloc[hdr_row]
    unit_row = raw.iloc[hdr_row+1] if hdr_row+1 < len(raw) else pd.Series(dtype=object)
    cols = build_columns_from_two_rows(header_row, unit_row)
    data = raw.iloc[hdr_row+2:].reset_index(drop=True)
    if len(cols) != data.shape[1]:
        data = data.iloc[:, :len(cols)]
    data.columns = cols
    data.columns = [c.replace("\n","") if isinstance(c,str) else c for c in data.columns]

    keep = [k for k in [
        "製品№ (1)","製品名 (大福生地)","実際売単価","必要販売単価","損益分岐単価","必要単価",
        "外注費","原価（材料費）","粗利 (0)","月間製造数(個数）","月間売上 (0)","月間支払 (0)",
        "付加価値率","日産製造数（個数）","合計 (151)","付加価値","1分当り付加価値","時","分",
        "受注数当り付加価値/日 (0)","1分当り付加価値2 (0)"
    ] if k in data.columns]
    df = data[keep].copy()

    def to_float(x):
        try:
            if x in ["", None, np.nan]:
                return np.nan
            return float(str(x).replace(",", ""))
        except:
            return np.nan

    for col in df.columns:
        if col not in ["製品№ (1)","製品名 (大福生地)"]:
            df[col] = df[col].map(to_float)

    rename_map = {
        "製品№ (1)": "product_no",
        "製品名 (大福生地)": "product_name",
        "実際売単価": "actual_unit_price",
        "原価（材料費）": "material_unit_cost",
        "日産製造数（個数）": "daily_qty",
        "分": "minutes_per_unit",
        "合計 (151)": "daily_total_minutes",
        "付加価値": "daily_va",
        "1分当り付加価値": "va_per_min",
        "必要販売単価": "required_selling_price_excel",
        "損益分岐単価": "be_unit_price_excel",
        "必要単価": "req_va_unit_price_excel",
        "粗利 (0)": "gp_per_unit_excel",
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # Core fields
    df["gp_per_unit"] = df.get("actual_unit_price", np.nan) - df.get("material_unit_cost", np.nan)

    # Safe compute minutes_per_unit
    if "minutes_per_unit" not in df.columns:
        df["minutes_per_unit"] = np.nan
    numer = series_or_nan(df, "daily_total_minutes")
    denom = series_or_nan(df, "daily_qty").replace({0: np.nan})
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_mpu = numer / denom
    df["minutes_per_unit"] = df["minutes_per_unit"].fillna(computed_mpu)

    # Safe compute daily_total_minutes
    if "daily_total_minutes" not in df.columns:
        df["daily_total_minutes"] = np.nan
    mpu = series_or_nan(df, "minutes_per_unit")
    qty = series_or_nan(df, "daily_qty")
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_total = mpu * qty
    df["daily_total_minutes"] = df["daily_total_minutes"].fillna(computed_total)

    # daily_va
    if "daily_va" not in df.columns:
        df["daily_va"] = np.nan
    gpu = series_or_nan(df, "gp_per_unit")
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_va = gpu * qty
    df["daily_va"] = df["daily_va"].fillna(computed_va)

    # va_per_min
    if "va_per_min" not in df.columns:
        df["va_per_min"] = np.nan
    total_min = series_or_nan(df, "daily_total_minutes").replace({0: np.nan})
    with np.errstate(divide='ignore', invalid='ignore'):
        computed_vapm = df["daily_va"] / total_min
    df["va_per_min"] = df["va_per_min"].fillna(computed_vapm)

    df = df[~(df.get("product_name").isna() & df.get("actual_unit_price").isna())].reset_index(drop=True)
    return df

def compute_results(df_products: pd.DataFrame, break_even_rate: float, required_rate: float) -> pd.DataFrame:
    df = df_products.copy()
    be_rate = 0.0 if break_even_rate is None else float(break_even_rate)
    req_rate = 0.0 if required_rate is None else float(required_rate)

    mpu = series_or_nan(df, "minutes_per_unit")
    df["be_va_unit_price"] = mpu * be_rate
    df["req_va_unit_price"] = mpu * req_rate
    df["required_selling_price"] = df.get("material_unit_cost") + df["req_va_unit_price"]
    df["price_gap_vs_required"] = df.get("actual_unit_price") - df["required_selling_price"]
    df["rate_gap_vs_required"] = df.get("va_per_min") - req_rate
    df["meets_required_rate"] = df["rate_gap_vs_required"] >= 0
    df["rate_class"] = df["rate_gap_vs_required"].apply(classify_rate_gap)
    out_cols = [
        "product_no","product_name",
        "actual_unit_price","material_unit_cost",
        "minutes_per_unit","daily_qty","daily_total_minutes",
        "gp_per_unit","daily_va","va_per_min",
        "be_va_unit_price","req_va_unit_price","required_selling_price",
        "price_gap_vs_required","rate_gap_vs_required","meets_required_rate","rate_class"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]

# ============== UI ==============
st.title("製品賃率計算（分/個 × 賃率）管理アプリ")

st.sidebar.header("データソース")
default_path = "data/sample.xlsx"
file = st.sidebar.file_uploader("Excelをアップロード（未指定ならサンプルを使用）", type=["xlsx"])
if file is None:
    st.sidebar.info("サンプルデータを使用します。")
    xls = read_excel_safely(default_path)
else:
    xls = read_excel_safely(file)
if xls is None:
    st.stop()

params = parse_hyochin(xls)
df_products_raw = parse_products(xls, sheet_name="R6.12")

st.sidebar.header("標準賃率（標賃）")
be_rate = st.sidebar.number_input("損益分岐賃率（円/分）", value=float(params.get("break_even_rate") or 0.0), step=0.001, format="%.6f")
req_rate = st.sidebar.number_input("必要賃率（円/分）", value=float(params.get("required_rate") or 0.0), step=0.001, format="%.6f")

with st.expander("賃率の根拠（参考）", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("固定費（計）", f"{params.get('fixed_total'):,}" if params.get('fixed_total') else "-")
    c2.metric("必要利益（計）", f"{params.get('required_profit_total'):,}" if params.get('required_profit_total') else "-")
    c3.metric("年間標準稼働時間（分）", f"{params.get('annual_minutes'):,}" if params.get('annual_minutes') else "-")

if df_products_raw.empty:
    st.warning("製品データが見つかりません。")
    st.stop()

df_results = compute_results(df_products_raw, be_rate, req_rate)

st.sidebar.header("分類フィルタ")
class_options = df_results["rate_class"].unique().tolist()
selected_classes = st.sidebar.multiselect("達成分類で絞り込み", class_options, default=class_options)
df_display = df_results[df_results["rate_class"].isin(selected_classes)].sort_values("rate_gap_vs_required")

colA, colB, colC, colD = st.columns(4)
colA.metric("必要賃率 (円/分)", f"{req_rate:,.3f}")
colB.metric("損益分岐賃率 (円/分)", f"{be_rate:,.3f}")
ach_rate = (df_display["meets_required_rate"].mean()*100.0) if len(df_display)>0 else 0.0
colC.metric("必要賃率達成SKU比率", f"{ach_rate:,.1f}%")
avg_vapm = df_display["va_per_min"].replace([np.inf,-np.inf], np.nan).dropna().mean() if "va_per_min" in df_display else 0.0
colD.metric("平均1分当り付加価値", f"{avg_vapm:,.1f}")

st.subheader("達成状況の分析")
class_counts = df_display["rate_class"].value_counts()
st.bar_chart(class_counts)

st.subheader("SKU別 計算結果")
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
df_table = df_display.rename(columns=rename_map)
df_table = df_table[[c for c in ordered_cols if c in df_table.columns]]

def _style_row(row):
    color = "#d1ffd6" if row.get("必要賃率達成") else "#ffd1d1"
    return [f"background-color: {color}"] * len(row)

def _highlight_negative(v):
    try:
        return "color: red" if v < 0 else ""
    except:
        return ""

styled = (
    df_table.style
    .apply(_style_row, axis=1)
    .applymap(_highlight_negative, subset=["必要販売単価差額","必要賃率差"])
    .format({
        "実際売単価": "{:,.0f}",
        "材料原価": "{:,.0f}",
        "分/個": "{:,.3f}",
        "日産数": "{:,.0f}",
        "日産合計(分)": "{:,.1f}",
        "粗利/個": "{:,.0f}",
        "付加価値(日産)": "{:,.0f}",
        "付加価値/分": "{:,.3f}",
        "損益分岐付加価値単価": "{:,.2f}",
        "必要付加価値単価": "{:,.2f}",
        "必要販売単価": "{:,.2f}",
        "必要販売単価差額": "{:,.2f}",
        "必要賃率差": "{:,.3f}",
    })
)
st.dataframe(styled, use_container_width=True, height=600)

csv = df_table.to_csv(index=False).encode("utf-8-sig")
st.download_button("結果をCSVでダウンロード", data=csv, file_name="calc_results.csv", mime="text/csv")

st.subheader("個別SKUの詳細（原データ）")
opts = ["(未選択)"] + df_products_raw["product_name"].dropna().astype(str).unique().tolist()
sel = st.selectbox("製品選択", options=opts)
if sel != "(未選択)":
    st.write(df_products_raw[df_products_raw["product_name"] == sel])

st.caption("※サンプルは添付Excelの『標賃』『R6.12』を解析して賃率・分/個・日産などを計算しています。")
