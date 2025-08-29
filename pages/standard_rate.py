from __future__ import annotations

"""標準賃率計算ページ

Streamlit のマルチページアプリに追加される単独ページ。固定費と必要利益から
損益分岐賃率・必要賃率を計算し、感度分析およびエクスポート機能を提供する。
"""

from io import BytesIO
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

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


class Node(TypedDict, total=False):
    """計算ノードの系譜情報"""

    key: str
    label: str
    value: float
    formula: str
    depends_on: List[str]
    unit: str | None


def build_node(
    key: str,
    label: str,
    value: float,
    formula: str,
    depends_on: List[str],
    unit: str | None = None,
) -> Node:
    """Node を生成する補助関数"""

    return Node(
        key=key,
        label=label,
        value=float(value),
        formula=formula,
        depends_on=depends_on,
        unit=unit,
    )


@dataclass
class FormulaSpec:
    """計算式の仕様"""

    label: str
    formula: str
    depends_on: List[str]
    unit: str | None
    func: Callable[[Dict[str, Node], Params], float]


FORMULAS: Dict[str, FormulaSpec] = {
    "fixed_total": FormulaSpec(
        label="固定費計",
        formula="labor_cost + sga_cost",
        depends_on=["labor_cost", "sga_cost"],
        unit="円/年",
        func=lambda n, p: p["labor_cost"] + p["sga_cost"],
    ),
    "required_profit_total": FormulaSpec(
        label="必要利益計",
        formula="loan_repayment + tax_payment + future_business",
        depends_on=["loan_repayment", "tax_payment", "future_business"],
        unit="円/年",
        func=lambda n, p: p["loan_repayment"] + p["tax_payment"] + p["future_business"],
    ),
    "net_workers": FormulaSpec(
        label="正味直接工員数",
        formula="fulltime_workers + 0.75*part1_workers + part2_coefficient*part2_workers",
        depends_on=["fulltime_workers", "part1_workers", "part2_workers", "part2_coefficient"],
        unit="人",
        func=lambda n, p: p["fulltime_workers"]
        + 0.75 * p["part1_workers"]
        + p["part2_coefficient"] * p["part2_workers"],
    ),
    "minutes_per_day": FormulaSpec(
        label="1日当り稼働時間（分）",
        formula="daily_hours*60",
        depends_on=["daily_hours"],
        unit="分/日",
        func=lambda n, p: p["daily_hours"] * 60,
    ),
    "standard_daily_minutes": FormulaSpec(
        label="1日標準稼働分",
        formula="minutes_per_day*operation_rate",
        depends_on=["minutes_per_day", "operation_rate"],
        unit="分/日",
        func=lambda n, p: n["minutes_per_day"]["value"] * p["operation_rate"],
    ),
    "annual_minutes": FormulaSpec(
        label="年間標準稼働分",
        formula="net_workers*working_days*standard_daily_minutes",
        depends_on=["net_workers", "working_days", "standard_daily_minutes"],
        unit="分/年",
        func=lambda n, p: n["net_workers"]["value"]
        * p["working_days"]
        * n["standard_daily_minutes"]["value"],
    ),
    "break_even_rate": FormulaSpec(
        label="損益分岐賃率",
        formula="fixed_total/annual_minutes",
        depends_on=["fixed_total", "annual_minutes"],
        unit="円/分",
        func=lambda n, p: n["fixed_total"]["value"] / n["annual_minutes"]["value"],
    ),
    "required_rate": FormulaSpec(
        label="必要賃率",
        formula="(fixed_total + required_profit_total)/annual_minutes",
        depends_on=["fixed_total", "required_profit_total", "annual_minutes"],
        unit="円/分",
        func=lambda n, p: (n["fixed_total"]["value"] + n["required_profit_total"]["value"])
        / n["annual_minutes"]["value"],
    ),
    "daily_be_va": FormulaSpec(
        label="1日当り損益分岐付加価値",
        formula="fixed_total/working_days",
        depends_on=["fixed_total", "working_days"],
        unit="円/日",
        func=lambda n, p: n["fixed_total"]["value"] / p["working_days"],
    ),
    "daily_req_va": FormulaSpec(
        label="1日当り必要利益付加価値",
        formula="(fixed_total + required_profit_total)/working_days",
        depends_on=["fixed_total", "required_profit_total", "working_days"],
        unit="円/日",
        func=lambda n, p: (n["fixed_total"]["value"] + n["required_profit_total"]["value"])
        / p["working_days"],
    ),
}

# 計算順序
FORMULA_KEYS = list(FORMULAS.keys())


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


@st.cache_data
def load_from_excel(file, defaults: Params) -> Tuple[Params, List[str], pd.DataFrame]:
    """Excelファイルからラベル検索で値を抽出する

    Returns
    -------
    Tuple[Params, List[str], pd.DataFrame]
        読み込まれたパラメータ、未検出ラベル、ラベルとキーの対応表
    """

    df = pd.read_excel(file, sheet_name="標賃", header=None)
    params: Params = defaults.copy()
    missing: List[str] = []
    mapping: List[dict[str, str | float]] = []
    for label, key in LABEL_MAP.items():
        val = _find_value(df, label)
        if val is not None:
            params[key] = float(val)
            mapping.append({"label": label, "key": key, "value": float(val)})
        else:
            missing.append(label)
    map_df = pd.DataFrame(mapping)
    return params, missing, map_df


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
        if np.isnan(val):
            warnings.append(f"{k} が NaN のため既定値を使用しました。")
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

def compute_rates(params: Params) -> Tuple[Dict[str, Node], Results]:
    """前提値から各指標を計算する純関数

    Parameters
    ----------
    params: Params
        サニタイズ済み入力パラメータ

    Returns
    -------
    Tuple[Dict[str, Node], Results]
        Node 辞書と値のみの辞書（後方互換用）
    """

    nodes: Dict[str, Node] = {}
    for key in FORMULA_KEYS:
        spec = FORMULAS[key]
        value = spec.func(nodes, params)
        node = build_node(
            key=key,
            label=spec.label,
            value=value,
            formula=spec.formula,
            depends_on=spec.depends_on,
            unit=spec.unit,
        )
        nodes[key] = node

    flat: Results = Results({k: v["value"] for k, v in nodes.items()})
    return nodes, flat


def expand_dependencies(nodes: Dict[str, Node]) -> Dict[str, List[str]]:
    """各ノードの依存関係を再帰的に展開し基底入力まで列挙"""

    cache: Dict[str, List[str]] = {}

    def _dfs(key: str) -> List[str]:
        if key in cache:
            return cache[key]
        node = nodes.get(key)
        if node is None:
            cache[key] = [key]
            return cache[key]
        deps: set[str] = set()
        for dep in node["depends_on"]:
            deps.update(_dfs(dep))
        cache[key] = sorted(deps)
        return cache[key]

    return {k: _dfs(k) for k in nodes}

def sensitivity_series(params: Params, key: str, grid: Iterable[float]) -> pd.Series:
    """指定パラメータを変化させたときの必要賃率を計算"""
    values: List[float] = []
    for val in grid:
        p = params.copy()
        p[key] = float(val)
        _, res = compute_rates(p)
        values.append(res["required_rate"])
    return pd.Series(values, index=list(grid))


def plot_sensitivity(params: Params):
    """各種パラメータの感度分析グラフ"""

    import matplotlib.pyplot as plt

    op_grid = np.linspace(0.5, 1.0, 11)
    s_op = sensitivity_series(params, "operation_rate", op_grid)

    worker_grid = np.arange(1, 11)
    s_worker = sensitivity_series(params, "fulltime_workers", worker_grid)

    days_grid = np.arange(200, 261, 10)
    s_days = sensitivity_series(params, "working_days", days_grid)

    factor_grid = np.linspace(0.8, 1.2, 9)
    fixed_vals: List[float] = []
    profit_vals: List[float] = []
    for f in factor_grid:
        p_fixed = params.copy()
        p_fixed["labor_cost"] *= f
        p_fixed["sga_cost"] *= f
        _, res_fixed = compute_rates(p_fixed)
        fixed_vals.append(res_fixed["required_rate"])

        p_profit = params.copy()
        p_profit["loan_repayment"] *= f
        p_profit["tax_payment"] *= f
        p_profit["future_business"] *= f
        _, res_profit = compute_rates(p_profit)
        profit_vals.append(res_profit["required_rate"])

    s_fixed = pd.Series(fixed_vals, index=factor_grid)
    s_profit = pd.Series(profit_vals, index=factor_grid)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].plot(s_op.index, s_op.values, label="必要賃率")
    axes[0].set_title("操業度→必要賃率")
    axes[0].set_xlabel("操業度")

    axes[1].plot(s_worker.index, s_worker.values, label="必要賃率")
    axes[1].set_title("正社員数→必要賃率")
    axes[1].set_xlabel("正社員数")

    axes[2].plot(s_days.index, s_days.values, label="必要賃率")
    axes[2].set_title("稼働日数→必要賃率")
    axes[2].set_xlabel("年間稼働日数")

    axes[3].plot(s_fixed.index, s_fixed.values, label="必要賃率")
    axes[3].set_title("固定費±20%→必要賃率")
    axes[3].set_xlabel("倍率")

    axes[4].plot(s_profit.index, s_profit.values, label="必要賃率")
    axes[4].set_title("必要利益±20%→必要賃率")
    axes[4].set_xlabel("倍率")

    for series, ax in zip([s_op, s_worker, s_days, s_fixed, s_profit], axes):
        ax.set_ylabel("円/分")
        ax.grid(True)
        ax.legend()
        ax.annotate(
            f"{series.values[-1]:.3f}",
            xy=(series.index[-1], series.values[-1]),
            textcoords="offset points",
            xytext=(0, -10),
            ha="center",
        )

    fig.tight_layout()
    return fig


def generate_pdf(
    nodes: Dict[str, Node],
    deps_map: Dict[str, List[str]],
    fig,
) -> bytes:
    """計算結果を1ページPDFにまとめる"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Table, TableStyle

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "標準賃率計算サマリー")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"損益分岐賃率（円/分）: {nodes['break_even_rate']['value']:.3f}")
    y -= 15
    c.drawString(40, y, f"必要賃率（円/分）: {nodes['required_rate']['value']:.3f}")
    y -= 15
    c.drawString(40, y, f"年間標準稼働時間（分）: {nodes['annual_minutes']['value']:.1f}")
    y -= 15
    c.drawString(40, y, f"正味直接工員数合計: {nodes['net_workers']['value']:.2f}")
    y -= 25
    top_keys = [
        "break_even_rate",
        "required_rate",
        "annual_minutes",
        "fixed_total",
        "required_profit_total",
    ]
    table_data = [["項目", "値", "式", "依存要素"]]
    for k in top_keys:
        n = nodes[k]
        table_data.append(
            [n["label"], f"{n['value']:,}", n["formula"], ", ".join(deps_map[k])]
        )
    tbl = Table(table_data, colWidths=[120, 80, 150, 150])
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
    from streamlit_js_eval import streamlit_js_eval

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
    if "highlight" not in st.session_state:
        st.session_state.highlight = []
    placeholders: Dict[str, DeltaGenerator] = {}

    st.sidebar.subheader("A) 必要固定費（円/年）")
    params["labor_cost"] = st.sidebar.number_input(
        "労務費", value=params["labor_cost"], step=1.0, format="%.0f", min_value=0.0
    )
    if "labor_cost" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["labor_cost"] = st.sidebar.empty()
    params["sga_cost"] = st.sidebar.number_input(
        "販管費", value=params["sga_cost"], step=1.0, format="%.0f", min_value=0.0
    )
    if "sga_cost" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["sga_cost"] = st.sidebar.empty()

    st.sidebar.subheader("B) 必要利益（円/年）")
    params["loan_repayment"] = st.sidebar.number_input(
        "借入返済（年）",
        value=params["loan_repayment"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    if "loan_repayment" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["loan_repayment"] = st.sidebar.empty()
    params["tax_payment"] = st.sidebar.number_input(
        "納税・納付",
        value=params["tax_payment"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    if "tax_payment" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["tax_payment"] = st.sidebar.empty()
    params["future_business"] = st.sidebar.number_input(
        "未来事業費",
        value=params["future_business"],
        step=1.0,
        format="%.0f",
        min_value=0.0,
    )
    if "future_business" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["future_business"] = st.sidebar.empty()

    st.sidebar.subheader("C) 工数前提")
    params["fulltime_workers"] = st.sidebar.number_input(
        "正社員：人数",
        value=params["fulltime_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    if "fulltime_workers" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["fulltime_workers"] = st.sidebar.empty()
    st.sidebar.caption("労働係数=1.00")
    params["part1_workers"] = st.sidebar.number_input(
        "準社員①：人数",
        value=params["part1_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    if "part1_workers" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["part1_workers"] = st.sidebar.empty()
    st.sidebar.caption("準社員① 労働係数=0.75")
    params["part2_workers"] = st.sidebar.number_input(
        "準社員②：人数",
        value=params["part2_workers"],
        step=1.0,
        format="%.2f",
        min_value=0.0,
    )
    if "part2_workers" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["part2_workers"] = st.sidebar.empty()
    params["part2_coefficient"] = st.sidebar.slider(
        "準社員②：労働係数",
        value=float(params["part2_coefficient"]),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )
    if "part2_coefficient" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["part2_coefficient"] = st.sidebar.empty()

    params["working_days"] = st.sidebar.number_input(
        "年間稼働日数（日）",
        value=params["working_days"],
        step=1.0,
        format="%.0f",
        min_value=1.0,
    )
    if "working_days" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["working_days"] = st.sidebar.empty()
    params["daily_hours"] = st.sidebar.number_input(
        "1日当り稼働時間（時間）",
        value=params["daily_hours"],
        step=0.1,
        format="%.2f",
        min_value=0.1,
    )
    if "daily_hours" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["daily_hours"] = st.sidebar.empty()
    params["operation_rate"] = st.sidebar.slider(
        "1日当り操業度",
        value=float(params["operation_rate"]),
        min_value=0.5,
        max_value=1.0,
        step=0.01,
    )
    if "operation_rate" in st.session_state.highlight:
        st.sidebar.info("← この指標が影響します")
    placeholders["operation_rate"] = st.sidebar.empty()

    st.sidebar.subheader("D) ファイル取込（任意）")
    uploaded = st.sidebar.file_uploader("標準賃率計算シート.xlsx", type="xlsx")
    params_before = params.copy()
    before_sanitized, _ = sanitize_params(params_before)
    _, before_results = compute_rates(before_sanitized)
    if uploaded is not None:
        try:
            loaded_params, missing, mapping_df = load_from_excel(uploaded, params)
            params.update(loaded_params)
            st.sidebar.dataframe(mapping_df, use_container_width=True)
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

    nodes, results = compute_rates(params)
    deps_map = expand_dependencies(nodes)

    if uploaded is not None:
        diff = results["required_rate"] - before_results["required_rate"]
        diff_df = pd.DataFrame(
            [{"指標": "required_rate", "before": before_results["required_rate"], "after": results["required_rate"], "差分": diff}]
        )
        st.sidebar.dataframe(diff_df)

    reverse_index: Dict[str, List[str]] = defaultdict(list)
    for key, deps in deps_map.items():
        for dep in deps:
            reverse_index[dep].append(key)

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
                cat_map.get(k, ""),
                n["label"],
                n["value"],
                n.get("unit", ""),
                n["formula"],
                ", ".join(deps_map[k]),
            )
            for k, n in nodes.items()
        ],
        columns=["区分", "項目", "値", "単位", "式", "依存要素"],
    )
    event = st.dataframe(
        df_break,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    if event and event.get("selection"):
        rows = event["selection"].get("rows", [])
        if rows:
            idx = rows[0]
            deps = df_break.iloc[idx]["依存要素"].split(", ") if df_break.iloc[idx]["依存要素"] else []
            st.session_state.highlight = [d for d in deps if d]

    st.subheader("感度分析")
    fig = plot_sensitivity(params)
    st.pyplot(fig)

    df_csv = pd.DataFrame(
        [
            {
                **n,
                "depends_on": ",".join(deps_map[k]),
            }
            for k, n in nodes.items()
        ]
    )
    csv = df_csv.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("CSVエクスポート", data=csv, file_name="standard_rate.csv", mime="text/csv")

    pdf_bytes = generate_pdf(nodes, deps_map, fig)
    st.download_button(
        "PDFエクスポート",
        data=pdf_bytes,
        file_name="standard_rate_summary.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()
