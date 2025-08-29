from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# --------- Column definitions (PDF schema) ---------
# Mapping of internal field names to display label/units
PRODUCT_SCHEMA: Dict[str, Dict[str, str]] = {
    "product_no": {"label": "製品№", "unit": None},
    "product_name": {"label": "製品名", "unit": None},
    "actual_unit_price": {"label": "実際売単価", "unit": "円"},
    "required_selling_price_excel": {"label": "必要販売単価", "unit": "円"},
    "be_unit_price_excel": {"label": "損益分岐単価", "unit": "円"},
    "req_va_unit_price_excel": {"label": "必要単価", "unit": "円"},
    "subcontract_cost": {"label": "外注費", "unit": "円"},
    "material_unit_cost": {"label": "原価（材料費）", "unit": "円"},
    "gp_per_unit_excel": {"label": "粗利", "unit": "円"},
    "monthly_qty": {"label": "月間製造数(個数)", "unit": "個"},
    "monthly_sales": {"label": "月間売上", "unit": "円"},
    "monthly_payments": {"label": "月間支払", "unit": "円"},
    "va_ratio": {"label": "付加価値率", "unit": "%"},
    "daily_qty": {"label": "日産製造数（個数）", "unit": "個"},
    "minutes_per_unit": {"label": "分/個", "unit": "分"},
    "daily_total_minutes": {"label": "日産合計(分)", "unit": "分"},
    "daily_va": {"label": "付加価値", "unit": "円"},
    "va_per_min": {"label": "1分当り付加価値", "unit": "円"},
    "va_per_order_per_day": {"label": "受注数当り付加価値/日", "unit": "円"},
    "va_per_min2": {"label": "1分当り付加価値2", "unit": "円"},
}

# Manufacturing process columns to be summed into minutes_per_unit
PROCESS_COLS: List[str] = [
    "準備","原材料調整","生地調整","熱工程","熱工程2","熱工程3","充填",
    "成型加工","成型加工2","熱工程4","前加工","前加工2","機械準備",
    "機械出し","機械分解","機械組立","冷却","片付け","準備2","非加熱材料調整",
    "熱工程5","冷却2","手細工加工等","手細工加工等2","仕上げ加工",
    "熱加工","冷加工","包装","片付け2",
]

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

def parse_hyochin(xls: pd.ExcelFile) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """『標賃』シートから諸元を抽出し、賃率を再計算"""
    from standard_rate_core import DEFAULT_PARAMS, sanitize_params, compute_rates

    warnings: List[str] = []
    try:
        df = pd.read_excel(xls, sheet_name="標賃", header=None)
    except Exception as e:
        warnings.append(f"シート『標賃』が読めません: {e}")
        return {}, DEFAULT_PARAMS.copy(), warnings

    def find_value(col1_kw: str | None = None, col2_kw: str | None = None) -> Optional[float]:
        mask = pd.Series(True, index=df.index)
        if col1_kw:
            mask &= df.iloc[:, 1].astype(str).str.contains(col1_kw, na=False)
        if col2_kw:
            mask &= df.iloc[:, 2].astype(str).str.contains(col2_kw, na=False)
        rows = df[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        for x in row[::-1]:
            try:
                return float(x)
            except Exception:
                continue
        return None

    extracted = {
        "labor_cost": find_value("労務費"),
        "sga_cost": find_value("販管費"),
        "loan_repayment": find_value("借入返済"),
        "tax_payment": find_value("納税"),
        "future_business": find_value("未来事業費"),
        "fulltime_workers": find_value("直接工員数", "正社員"),
        "part1_workers": find_value(col2_kw="準社員➀"),
        "part2_workers": find_value(col2_kw="準社員②"),
        "working_days": find_value("年間稼働日数"),
        "daily_hours": find_value("1日当り稼働時間"),
        "operation_rate": find_value("1日当り操業度"),
    }

    # part2 coefficient (row after 準社員②)
    part2_coef = None
    rows = df[df.iloc[:, 2].astype(str).str.contains("準社員②", na=False)]
    if not rows.empty:
        idx = rows.index[0]
        for i in range(idx + 1, min(idx + 4, len(df))):
            if "労働係数" in str(df.iloc[i, 2]):
                try:
                    part2_coef = float(df.iloc[i, 3])
                except Exception:
                    pass
                break
    extracted["part2_coefficient"] = part2_coef

    sr_params: Dict[str, float] = {}
    for k, default in DEFAULT_PARAMS.items():
        v = extracted.get(k)
        if v is None:
            warnings.append(f"{k} をExcelから取得できませんでした。既定値を使用します。")
            sr_params[k] = default
        else:
            sr_params[k] = v

    sr_params, warn2 = sanitize_params(sr_params)
    warnings.extend(warn2)
    _, flat = compute_rates(sr_params)
    params = {k: flat[k] for k in [
        "fixed_total",
        "required_profit_total",
        "annual_minutes",
        "break_even_rate",
        "required_rate",
        "daily_be_va",
        "daily_req_va",
    ]}
    return params, sr_params, warnings

def parse_products(xls: pd.ExcelFile, sheet_name: str = "R6.12") -> Tuple[pd.DataFrame, List[str]]:
    """『R6.12』の製品マスタをPDF列スキーマに沿って構造化"""
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
    # 単位行はヘッダ行の前後どちらかに存在する可能性がある
    unit_row = pd.Series(dtype=object)
    unit_row_idx = hdr_row + 1
    if hdr_row - 1 >= 0:
        candidate = raw.iloc[hdr_row - 1]
        if candidate.dropna().astype(str).str.contains(r"[円個分％]").any():
            unit_row = candidate
            unit_row_idx = hdr_row - 1
    if unit_row.empty and hdr_row + 1 < len(raw):
        unit_row = raw.iloc[hdr_row + 1]
        unit_row_idx = hdr_row + 1
    cols = build_columns_from_two_rows(header_row, unit_row)
    data = raw.iloc[unit_row_idx + 1 :].reset_index(drop=True)
    if len(cols) != data.shape[1]:
        data = data.iloc[:, : len(cols)]
    data.columns = [c.replace("\n", "") if isinstance(c, str) else c for c in cols]

    df = data.copy()

    def to_float(x):
        try:
            if x in ["", None, np.nan]:
                return np.nan
            return float(str(x).replace(",", ""))
        except Exception:
            return np.nan

    # Rename columns based on labels defined in PRODUCT_SCHEMA
    rename_map: Dict[str, str] = {}
    for key, meta in PRODUCT_SCHEMA.items():
        label = meta["label"]
        matches = [c for c in df.columns if str(c).startswith(label)]
        for m in matches:
            rename_map[m] = key

    df = df.rename(columns=rename_map)

    for col in df.columns:
        if col not in ["product_no", "product_name"]:
            df[col] = df[col].map(to_float)

    # Ensure process columns exist and are numeric
    for pcol in PROCESS_COLS:
        if pcol not in df.columns:
            df[pcol] = 0.0
        df[pcol] = df[pcol].map(to_float)

    # minutes_per_unit from process columns
    df["minutes_per_unit"] = df[PROCESS_COLS].sum(axis=1, skipna=True)

    df = df[~(df.get("product_name").isna() & df.get("actual_unit_price").isna())].reset_index(drop=True)
    return df, warnings

# --------------- Core compute ---------------
def compute_results(df_products: pd.DataFrame, break_even_rate: float, required_rate: float) -> pd.DataFrame:
    df = df_products.copy()
    # Excel からの読み込みや二重の計算処理により、同名の列が
    # 複数存在することがある。pandas の演算は重複した列ラベルを
    # 含むデータフレームに対して reindex を行う際に ValueError を
    # 投げるため、ここで重複列を除去しておく。
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    be_rate = 0.0 if break_even_rate is None else float(break_even_rate)
    req_rate = 0.0 if required_rate is None else float(required_rate)
    # recompute core metrics from raw columns
    actual = series_or_nan(df, "actual_unit_price")
    material = series_or_nan(df, "material_unit_cost")
    subcontract = series_or_nan(df, "subcontract_cost")
    qty = series_or_nan(df, "daily_qty")
    mpu = series_or_nan(df, "minutes_per_unit")

    df["gp_per_unit"] = actual - material - subcontract
    df["daily_total_minutes"] = mpu * qty
    df["daily_va"] = df["gp_per_unit"] * qty
    with np.errstate(divide='ignore', invalid='ignore'):
        df["va_per_min"] = df["daily_va"] / df["daily_total_minutes"].replace({0: np.nan})

    df["be_va_unit_price"] = mpu * be_rate
    df["req_va_unit_price"] = mpu * req_rate
    df["required_selling_price"] = material + subcontract + df["req_va_unit_price"]
    df["price_gap_vs_required"] = actual - df["required_selling_price"]
    df["rate_gap_vs_required"] = df["va_per_min"] - req_rate
    df["meets_required_rate"] = df["rate_gap_vs_required"] >= 0
    df["rate_class"] = df["rate_gap_vs_required"].apply(classify_rate_gap)

    out_cols = [
        "product_no",
        "product_name",
        "actual_unit_price",
        "material_unit_cost",
        "subcontract_cost",
        "minutes_per_unit",
        "daily_qty",
        "daily_total_minutes",
        "gp_per_unit",
        "daily_va",
        "va_per_min",
        "be_va_unit_price",
        "req_va_unit_price",
        "required_selling_price",
        "price_gap_vs_required",
        "rate_gap_vs_required",
        "meets_required_rate",
        "rate_class",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols]
