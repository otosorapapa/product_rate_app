from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, TypedDict, Set

import numpy as np
import pandas as pd

# Reusable computational core extracted from the user's standard_rate.py

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

class Node(TypedDict, total=False):
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
    label: str
    formula: str
    depends_on: List[str]
    unit: str | None
    func: Callable[[Dict[str, Node], Dict[str, float]], float]

FORMULAS: Dict[str, FormulaSpec] = {
    "fixed_total": FormulaSpec(
        label="固定費計",
        formula="labor_cost + sga_cost",  # PDF: D5 = 労務費 + 販管費
        depends_on=["labor_cost", "sga_cost"],
        unit="円/年",
        func=lambda n, p: p["labor_cost"] + p["sga_cost"],
    ),
    "required_profit_total": FormulaSpec(
        label="必要利益計",
        formula="loan_repayment + tax_payment + future_business",  # PDF: D15 = 借入返済 + 納税・納付 + 未来事業費
        depends_on=["loan_repayment", "tax_payment", "future_business"],
        unit="円/年",
        func=lambda n, p: p["loan_repayment"] + p["tax_payment"] + p["future_business"],
    ),
    "net_workers": FormulaSpec(
        label="正味直接工員数",
        formula="fulltime_workers + 0.75*part1_workers + part2_coefficient*part2_workers",
        depends_on=["fulltime_workers", "part1_workers", "part2_workers", "part2_coefficient"],
        unit="人",
        func=lambda n, p: p["fulltime_workers"] + 0.75*p["part1_workers"] + p["part2_coefficient"]*p["part2_workers"],
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
        formula="net_workers*working_days*standard_daily_minutes",  # PDF: D33 = 正味直接工員数 * 年間稼働日数 * 1日標準稼働分
        depends_on=["net_workers", "working_days", "standard_daily_minutes"],
        unit="分/年",
        func=lambda n, p: n["net_workers"]["value"] * p["working_days"] * n["standard_daily_minutes"]["value"],
    ),
    "break_even_rate": FormulaSpec(
        label="損益分岐賃率",
        formula="fixed_total/annual_minutes",  # PDF: 損益分岐賃率 = D5/D33
        depends_on=["fixed_total", "annual_minutes"],
        unit="円/分",
        func=lambda n, p: n["fixed_total"]["value"] / n["annual_minutes"]["value"],
    ),
    "required_rate": FormulaSpec(
        label="必要賃率",
        formula="(fixed_total + required_profit_total)/annual_minutes",  # PDF: 必要賃率 = D15/D33
        depends_on=["fixed_total", "required_profit_total", "annual_minutes"],
        unit="円/分",
        func=lambda n, p: (n["fixed_total"]["value"] + n["required_profit_total"]["value"])/ n["annual_minutes"]["value"],
    ),
    "daily_be_va": FormulaSpec(
        label="1日当り損益分岐付加価値",
        formula="固定費計/稼働日数",  # PDF: =D5/D28
        depends_on=["fixed_total", "working_days"],
        unit="円/日",
        func=lambda n, p: n["fixed_total"]["value"] / p["working_days"],
    ),
    "daily_req_va": FormulaSpec(
        label="1日当り必要利益付加価値",
        formula="(固定費計 + 必要利益計)/稼働日数",  # PDF: =D15/D28
        depends_on=["fixed_total", "required_profit_total", "working_days"],
        unit="円/日",
        func=lambda n, p: (n["fixed_total"]["value"] + n["required_profit_total"]["value"])/ p["working_days"],
    ),
}

FORMULA_KEYS = list(FORMULAS.keys())

def sanitize_params(params: Dict[str, float]) -> tuple[Dict[str, float], list[str]]:
    sanitized = {**DEFAULT_PARAMS}
    warnings: list[str] = []
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
    net = sanitized["fulltime_workers"] + 0.75*sanitized["part1_workers"] + sanitized["part2_coefficient"]*sanitized["part2_workers"]
    if net <= 0:
        warnings.append("正味直接工員数が0以下のため1に補正しました。")
        sanitized["fulltime_workers"] = 1.0
    return sanitized, warnings

def compute_rates(params: Dict[str, float]):
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
    flat = {k: v["value"] for k, v in nodes.items()}
    return nodes, flat

def build_reverse_index(nodes: Dict[str, Node]) -> Dict[str, list[str]]:
    dep_map: Dict[str, set[str]] = { key: set(node["depends_on"]) for key, node in nodes.items() }
    changed = True
    while changed:
        changed = False
        for key, deps in dep_map.items():
            extra = set()
            for d in list(deps):
                extra |= dep_map.get(d, set())
            if not extra.issubset(deps):
                deps |= extra
                changed = True
    reverse: Dict[str, list[str]] = {}
    for node_key, deps in dep_map.items():
        for dep in deps:
            reverse.setdefault(dep, []).append(node_key)
    return reverse

def sensitivity_series(params: Dict[str, float], key: str, grid):
    values: List[float] = []
    for val in grid:
        p = {**params}
        p[key] = float(val)
        _, res = compute_rates(p)
        values.append(res["required_rate"])
    import pandas as pd
    return pd.Series(values, index=list(grid))

def plot_sensitivity(params: Dict[str, float]):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    op_grid = np.linspace(0.5, 1.0, 11)
    s_op = sensitivity_series(params, "operation_rate", op_grid)
    worker_grid = np.arange(1, 11)
    s_worker = sensitivity_series(params, "fulltime_workers", worker_grid)
    days_grid = np.arange(200, 261, 10)
    s_days = sensitivity_series(params, "working_days", days_grid)
    factor_grid = np.linspace(0.8, 1.2, 9)
    fixed_vals = []; profit_vals = []
    for f in factor_grid:
        p_fixed = {**params}
        p_fixed["labor_cost"] *= f; p_fixed["sga_cost"] *= f
        _, res_fixed = compute_rates(p_fixed); fixed_vals.append(res_fixed["required_rate"])
        p_profit = {**params}
        p_profit["loan_repayment"] *= f; p_profit["tax_payment"] *= f; p_profit["future_business"] *= f
        _, res_profit = compute_rates(p_profit); profit_vals.append(res_profit["required_rate"])
    s_fixed = pd.Series(fixed_vals, index=factor_grid)
    s_profit = pd.Series(profit_vals, index=factor_grid)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].plot(s_op.index, s_op.values, label="必要賃率"); axes[0].set_title("操業度→必要賃率"); axes[0].set_xlabel("操業度")
    axes[1].plot(s_worker.index, s_worker.values, label="必要賃率"); axes[1].set_title("正社員数→必要賃率"); axes[1].set_xlabel("正社員数")
    axes[2].plot(s_days.index, s_days.values, label="必要賃率"); axes[2].set_title("稼働日数→必要賃率"); axes[2].set_xlabel("年間稼働日数")
    axes[3].plot(s_fixed.index, s_fixed.values, label="必要賃率"); axes[3].set_title("固定費±20%→必要賃率"); axes[3].set_xlabel("倍率")
    axes[4].plot(s_profit.index, s_profit.values, label="必要賃率"); axes[4].set_title("必要利益±20%→必要賃率"); axes[4].set_xlabel("倍率")
    for series, ax in zip([s_op, s_worker, s_days, s_fixed, s_profit], axes):
        ax.set_ylabel("円/分"); ax.grid(True); ax.legend()
        ax.annotate(f"{series.values[-1]:.3f}", xy=(series.index[-1], series.values[-1]), textcoords="offset points", xytext=(0, -10), ha="center")
    fig.tight_layout()
    return fig

def generate_pdf(nodes: Dict[str, Node], fig) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Table, TableStyle
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica-Bold", 16); c.drawString(40, y, "標準賃率計算サマリー"); y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"損益分岐賃率（円/分）: {nodes['break_even_rate']['value']:.3f}"); y -= 15
    c.drawString(40, y, f"必要賃率（円/分）: {nodes['required_rate']['value']:.3f}"); y -= 15
    c.drawString(40, y, f"年間標準稼働時間（分）: {nodes['annual_minutes']['value']:.1f}"); y -= 15
    c.drawString(40, y, f"正味直接工員数合計: {nodes['net_workers']['value']:.2f}"); y -= 25
    top_keys = ["break_even_rate", "required_rate", "annual_minutes", "fixed_total", "required_profit_total"]
    table_data = [["項目", "値", "式", "依存要素"]]
    for k in top_keys:
        n = nodes[k]
        table_data.append([n["label"], f"{n['value']:,}", n["formula"], ", ".join(n["depends_on"])])
    tbl = Table(table_data, colWidths=[120, 80, 150, 150])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    tw, th = tbl.wrap(0, 0); tbl.drawOn(c, 40, y - th); y = y - th - 20
    img_buf = BytesIO(); fig.savefig(img_buf, format="png", bbox_inches="tight"); img_buf.seek(0); img = ImageReader(img_buf)
    c.drawImage(img, 40, 40, width=width - 80, preserveAspectRatio=True, mask='auto')
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()
