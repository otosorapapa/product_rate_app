from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# --------------- Low-level helpers ---------------
def _clean(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\n", "").strip()

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

# --------------- Excel parsing ---------------
def read_excel_safely(path_or_bytes) -> Optional[pd.ExcelFile]:
    try:
        xls = pd.ExcelFile(path_or_bytes, engine="openpyxl")
        return xls
    except Exception:
        return None

def parse_hyochin(xls: pd.ExcelFile) -> Tuple[Dict[str, Any], List[str]]:
    """『標賃』シートから主要パラメータを抽出"""
    warnings: List[str] = []
    params: Dict[str, Any] = dict(
        labor_cost=None, sgna=None, fixed_total=None,
        required_profit_total=None, annual_minutes=None,
        break_even_rate=None, required_rate=None
    )
    try:
        df = pd.read_excel(xls, sheet_name="標賃", header=None)
    except Exception as e:
        warnings.append(f"シート『標賃』が読めません: {e}")
        return params, warnings

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
            except Exception:
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
                except Exception:
                    pass

    annual_minutes = None
    rows = df[(df.iloc[:,1].astype(str).str.contains("年間標準稼働時間（分）", na=False))]
    if not rows.empty:
        try:
            annual_minutes = float(rows.iloc[0,3])
        except Exception:
            annual_minutes = None

    be_rate = None; req_rate = None
    rows_be = df[(df.iloc[:,1].astype(str).str.contains("損益分岐賃率", na=False))]
    if not rows_be.empty:
        try:
            be_rate = float(rows_be.iloc[0,3])
        except Exception:
            pass
    rows_req = df[(df.iloc[:,1].astype(str).str.contains("必要賃率", na=False))]
    if not rows_req.empty:
        try:
            req_rate = float(rows_req.iloc[0,3])
        except Exception:
            pass

    if (be_rate is None or req_rate is None) and fixed_total and annual_minutes:
        try:
            if be_rate is None:
                be_rate = fixed_total / annual_minutes
            if req_rate is None and required_profit_total:
                req_rate = (fixed_total + required_profit_total) / annual_minutes
        except Exception:
            pass

    params.update(dict(
        labor_cost=labor,
        sgna=sgna,
        fixed_total=fixed_total,
        required_profit_total=required_profit_total,
        annual_minutes=annual_minutes,
        break_even_rate=be_rate,
        required_rate=req_rate,
    ))
    return params, warnings

def parse_products(xls: pd.ExcelFile, sheet_name: str="R6.12") -> Tuple[pd.DataFrame, List[str]]:
    """『R6.12』の製品マスタを構造化"""
    warnings: List[str] = []
    try:
        raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
        warnings.append(f"シート『{sheet_name}』が読めません: {e}")
        return pd.DataFrame(), warnings

    hdr_row = find_header_row(raw, "製品№")
    if hdr_row < 0:
        warnings.append("『製品№』行が見つかりません。")
        return pd.DataFrame(), warnings

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
        except Exception:
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
    return df, warnings

# --------------- Core compute ---------------
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
