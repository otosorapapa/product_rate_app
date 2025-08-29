from __future__ import annotations

"""標準賃率計算ページ

Streamlit のマルチページアプリに追加される単独ページ。固定費と必要利益から
損益分岐賃率・必要賃率を計算し、感度分析およびエクスポート機能を提供する。
"""

from io import BytesIO
import json
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

# ================================================================
# デフォルト値とラベルマップ
# ================================================================
DEFAULT_PARAMS: Dict[str, float] = {
    "labor_cost": 16_829_175.0,
    "sga_cost": 40_286_204.0,
    "loan_repayment": 4_000_000.0,
    "tax_payment": 3_797_500.0,
    "future_business": 2_000_000.0,
    "fulltime_workers": 4.0,
    "part1_workers": 2.0,
    "part2_workers": 0.0,
    "part2_coefficient": 0.0,
    "working_days": 236.0,
    "daily_hours": 8.68,
    "operation_rate": 0.75,
}

LABEL_MAP: Dict[str, str] = {
    "労務費": "labor_cost",
    "販管費": "sga_cost",
    "借入返済": "loan_repayment",
    "借入返済年": "loan_repayment",
    "納税納付": "tax_payment",
    "未来事業費": "future_business",
    "直接工員数/正社員": "fulltime_workers",
    "正社員": "fulltime_workers",
    "準社員①": "part1_workers",
    "準社員1": "part1_workers",
    "準社員②": "part2_workers",
    "準社員2": "part2_workers",
    "労働係数": "part2_coefficient",
    "年間稼働日数": "working_days",
    "1日当り稼働時間時間": "daily_hours",
    "1日当り稼働時間": "daily_hours",
    "1日当たり稼働時間": "daily_hours",
    "1日当り操業度": "operation_rate",
    "1日当たり操業度": "operation_rate",
}

class Params(Dict[str, float]):
    """入力パラメータの辞書"""


class Results(Dict[str, float]):
    """計算結果の辞書"""


# ================================================================
# ユーティリティ関数
# ================================================================

def _norm(s: str | float | int | None) -> str:
    """ラベル比較用の正規化"""
    if s is None:
        return ""
    s2 = unicodedata.normalize("NFKC", str(s))
    s2 = re.sub(r"[\s()（）]", "", s2)
    return s2


def _find_value(df: pd.DataFrame, label: str) -> float | None:
    """DataFrameからラベルに対応する数値を探索"""
    target = _norm(label)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            cell = df.iat[i, j]
            if isinstance(cell, str) and target in _norm(cell):
                for k in range(j + 1, df.shape[1]):
                    val = df.iat[i, k]
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
                for k in range(i + 1, df.shape[0]):
                    val = df.iat[k, j]
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
    return None


def load_from_excel(file, defaults: Params) -> Tuple[Params, List[str]]:
    """Excelファイルからラベル検索で値を抽出する"""
    df = pd.read_excel(file, sheet_name="標賃", header=None)
    params: Params = defaults.copy()
    missing: List[str] = []
    for label, key in LABEL_MAP.items():
        val = _find_value(df, label)
        if val is not None:
            params[key] = float(val)
        else:
            missing.append(label)
    return params, missing


def sanitize_params(params: Params) -> Tuple[Params, List[str]]:
    """負数や不正値を補正し警告メッセージを返す"""
    sanitized: Params = DEFAULT_PARAMS.copy()
    warnings: List[str] = []
    for k, default in DEFAULT_PARAMS.items():
        raw = params.get(k, default)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"{k} が数値でないため既定値を使用しました。")
            val = default
        if val < 0:
            warnings.append(f"{k} が負数のため0に補正しました。")
            val = 0.0
        sanitized[k] = val
    if sanitized["working_days"] <= 0:
        warnings.append("年間稼働日数が0以下のため1に補正しました。")
        sanitized["working_days"] = 1.0
    if sanitized["daily_hours"] <= 0:
        warnings.append("1日当り稼働時間が0以下のため1に補正しました。")
        sanitized["daily_hours"] = 1.0
    if sanitized["operation_rate"] <= 0:
        warnings.append("1日当り操業度が0以下のため0.01に補正しました。")
        sanitized["operation_rate"] = 0.01
    net = (
        sanitized["fulltime_workers"]
        + sanitized["part1_workers"] * 0.75
        + sanitized["part2_workers"] * sanitized["part2_coefficient"]
    )
    if net <= 0:
        warnings.append("正味直接工員数が0以下のため1に補正しました。")
        sanitized["fulltime_workers"] = 1.0
    return sanitized, warnings


def compute_rates(params: Params) -> Results:
    """前提値から損益分岐賃率・必要賃率を計算する"""
    labor = params["labor_cost"]
    sga = params["sga_cost"]
    loan = params["loan_repayment"]
    tax = params["tax_payment"]
    future = params["future_business"]
    fixed_total = labor + sga
    required_profit_total = loan + tax + future
    fixed_plus_required = fixed_total + required_profit_total
    net_workers = (
        params["fulltime_workers"]
        + params["part1_workers"] * 0.75
        + params["part2_workers"] * params["part2_coefficient"]
    )
    minutes_per_day = params["daily_hours"] * 60.0
    standard_daily_minutes = minutes_per_day * params["operation_rate"]
    annual_minutes = net_workers * params["working_days"] * standard_daily_minutes
    if annual_minutes <= 0:
        be_rate = np.nan
        req_rate = np.nan
    else:
        be_rate = fixed_total / annual_minutes
        req_rate = fixed_plus_required / annual_minutes
    daily_be_va = fixed_total / params["working_days"] if params["working_days"] > 0 else np.nan
    daily_req_va = (
        fixed_plus_required / params["working_days"]
        if params["working_days"] > 0
        else np.nan
    )
    return Results(
        fixed_total=fixed_total,
        required_profit_total=required_profit_total,
        fixed_plus_required=fixed_plus_required,
        net_workers=net_workers,
        minutes_per_day=minutes_per_day,
        standard_daily_minutes=standard_daily_minutes,
        annual_minutes=annual_minutes,
        daily_be_va=daily_be_va,
        daily_req_va=daily_req_va,
        break_even_rate=be_rate,
        required_rate=req_rate,
    )

def sensitivity_series(params: Params, key: str, grid: Iterable[float]) -> pd.Series:
    """指定パラメータを変化させたときの必要賃率を計算"""
    values: List[float] = []
    for val in grid:
        p = params.copy()
        p[key] = float(val)
        res = compute_rates(p)
        values.append(res["required_rate"])
    return pd.Series(values, index=list(grid))


def plot_sensitivity(params: Params) -> plt.Figure:
    """操業度・人員・稼働日数の感度分析グラフ"""
    op_grid = np.linspace(0.5, 1.0, 11)
    s_op = sensitivity_series(params, "operation_rate", op_grid)
    worker_grid = np.arange(1, 11)
    s_worker = sensitivity_series(params, "fulltime_workers", worker_grid)
    days_grid = np.arange(200, 261, 10)
    s_days = sensitivity_series(params, "working_days", days_grid)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(s_op.index, s_op.values)
    axes[0].set_title("操業度→必要賃率")
    axes[0].set_xlabel("操業度")
    axes[0].set_ylabel("円/分")
    axes[1].plot(s_worker.index, s_worker.values)
    axes[1].set_title("正社員数→必要賃率")
    axes[1].set_xlabel("正社員数")
    axes[2].plot(s_days.index, s_days.values)
    axes[2].set_title("稼働日数→必要賃率")
    axes[2].set_xlabel("年間稼働日数")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    return fig


def generate_pdf(params: Params, results: Results, fig: plt.Figure) -> bytes:
    """計算結果を1ページPDFにまとめる"""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "標準賃率計算サマリー")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"損益分岐賃率（円/分）: {results['break_even_rate']:.3f}")
    y -= 15
    c.drawString(40, y, f"必要賃率（円/分）: {results['required_rate']:.3f}")
    y -= 15
    c.drawString(40, y, f"年間標準稼働時間（分）: {results['annual_minutes']:.1f}")
    y -= 15
    c.drawString(40, y, f"正味直接工員数合計: {results['net_workers']:.2f}")
    y -= 25
    table_data = [
        ["項目", "値"],
        ["固定費計", f"{results['fixed_total']:,}"],
        ["必要利益計", f"{results['required_profit_total']:,}"],
        ["1日当り損益分岐付加価値", f"{results['daily_be_va']:,}"],
        ["1日当り必要利益付加価値", f"{results['daily_req_va']:,}"],
    ]
    tbl = Table(table_data, colWidths=[180, 180])
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    tw, th = tbl.wrap(0, 0)
    tbl.drawOn(c, 40, y - th)
    y = y - th - 20
    img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)
    img = ImageReader(img_buf)
    c.drawImage(img, 40, 40, width=width - 80, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="標準賃率計算", layout="wide")
    st.title("標準賃率計算（円/分）")

    if "sr_params" not in st.session_state:
        loaded = streamlit_js_eval(
            js_expressions="window.localStorage.getItem('standard_rate_params')",
            key="load_params",
        )
        if loaded:
            try:
                st.session_state.sr_params = {
                    **DEFAULT_PARAMS,
                    **json.loads(loaded),
                }
            except Exception:
                st.session_state.sr_params = DEFAULT_PARAMS.copy()
        else:
            st.session_state.sr_params = DEFAULT_PARAMS.copy()
    params: Params = st.session_state.sr_params.copy()

    st.sidebar.header("入力")
    st.sidebar.subheader("A) 必要固定費（円/年）")
    params["labor_cost"] = st.sidebar.number_input(
        "労務費", value=params["labor_cost"], step=1.0, format="%.0f", min_value=0.0
    )
    params["sga_cost"] = st.sidebar.number_input(
        "販管費", value=params["sga_cost"], step=1.0, format="%.0f", min_value=0.0
    )
    st.sidebar.caption(f"固定費計 = {params['labor_cost'] + params['sga_cost']:,}")

    st.sidebar.subheader("B) 必要利益（円/年）")
    params["loan_repayment"] = st.sidebar.number_input(
        "借入返済（年）",
        value=params["loan_repayment"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    params["tax_payment"] = st.sidebar.number_input(
        "納税・納付",
        value=params["tax_payment"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    params["future_business"] = st.sidebar.number_input(
        "未来事業費",
        value=params["future_business"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    st.sidebar.caption(
        f"必要利益計 = {params['loan_repayment'] + params['tax_payment'] + params['future_business']:,}"
    )

    st.sidebar.subheader("C) 工数前提")
    params["fulltime_workers"] = st.sidebar.number_input(
        "正社員：人数",
        value=params["fulltime_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    st.sidebar.caption("労働係数=1.00")
    params["part1_workers"] = st.sidebar.number_input(
        "準社員①：人数",
        value=params["part1_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    st.sidebar.caption("準社員① 労働係数=0.75")
    params["part2_workers"] = st.sidebar.number_input(
        "準社員②：人数",
        value=params["part2_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    params["part2_coefficient"] = st.sidebar.slider(
        "準社員②：労働係数",
        value=float(params["part2_coefficient"]),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    net_direct = (
        params["fulltime_workers"]
        + params["part1_workers"] * 0.75
        + params["part2_workers"] * params["part2_coefficient"]
    )
    st.sidebar.caption(f"正味直接工員数合計 = {net_direct:.2f}")

    params["working_days"] = st.sidebar.number_input(
        "年間稼働日数（日）",
        value=params["working_days"],
        step=1.0,
        format="%.0f",
        min_value=1.0,
    )
    params["daily_hours"] = st.sidebar.number_input(
        "1日当り稼働時間（時間）",
        value=params["daily_hours"],
        step=0.1,
        format="%.2f",
        min_value=0.1,
    )
    st.sidebar.caption(f"= {params['daily_hours']*60:.1f} 分")
    params["operation_rate"] = st.sidebar.slider(
        "1日当り操業度", value=float(params["operation_rate"]), min_value=0.5, max_value=1.0, step=0.01
    )

    st.sidebar.subheader("D) ファイル取込（任意）")
    uploaded = st.sidebar.file_uploader("標準賃率計算シート.xlsx", type="xlsx")
    if uploaded is not None:
        try:
            params_from_file, missing = load_from_excel(uploaded, params)
            params.update(params_from_file)
            if missing:
                st.sidebar.warning("未検出ラベル: " + ", ".join(missing))
        except Exception as e:
            st.sidebar.error(f"Excel読込エラー: {e}")

    params, warn_list = sanitize_params(params)
    for w in warn_list:
        st.sidebar.warning(w)
    st.session_state.sr_params = params
    streamlit_js_eval(
        js_expressions=f"window.localStorage.setItem('standard_rate_params', `{json.dumps(params, ensure_ascii=False)}`)",
        key="save_params",
    )

    results = compute_rates(params)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("損益分岐賃率（円/分）", f"{results['break_even_rate']:.3f}")
    c2.metric("必要賃率（円/分）", f"{results['required_rate']:.3f}")
    c3.metric("年間標準稼働時間（分）", f"{results['annual_minutes']:.0f}")
    c4.metric("正味直接工員数合計", f"{results['net_workers']:.2f}")

    breakdown_data = [
        ("固定費", "労務費", params["labor_cost"]),
        ("固定費", "販管費", params["sga_cost"]),
        ("固定費", "固定費計", results["fixed_total"]),
        ("必要利益", "借入返済（年）", params["loan_repayment"]),
        ("必要利益", "納税・納付", params["tax_payment"]),
        ("必要利益", "未来事業費", params["future_business"]),
        ("必要利益", "必要利益計", results["required_profit_total"]),
        ("工数前提", "正社員数", params["fulltime_workers"]),
        ("工数前提", "準社員①数", params["part1_workers"]),
        ("工数前提", "準社員②数", params["part2_workers"]),
        ("工数前提", "準社員②労働係数", params["part2_coefficient"]),
        ("工数前提", "正味直接工員数合計", results["net_workers"]),
        ("工数前提", "年間稼働日数", params["working_days"]),
        ("工数前提", "1日当り稼働時間（分）", results["minutes_per_day"]),
        ("工数前提", "1日当り操業度", params["operation_rate"]),
        ("付加価値", "1日当り損益分岐付加価値", results["daily_be_va"]),
        ("付加価値", "1日当り必要利益付加価値", results["daily_req_va"]),
    ]
    df_break = pd.DataFrame(breakdown_data, columns=["区分", "項目", "値"])
    st.subheader("ブレークダウン")
    st.dataframe(df_break.style.format({"値": "{:,.3f}"}), use_container_width=True)

    st.subheader("感度分析")
    fig = plot_sensitivity(params)
    st.pyplot(fig)

    df_row = pd.DataFrame([{**params, **results}])
    csv = df_row.to_csv(index=False).encode("utf-8-sig")
    st.download_button("CSVエクスポート", data=csv, file_name="standard_rate.csv", mime="text/csv")

    pdf_bytes = generate_pdf(params, results, fig)
    st.download_button(
        "PDFエクスポート",
        data=pdf_bytes,
        file_name="standard_rate_summary.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()
