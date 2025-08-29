import math
from pages.standard_rate import (
    DEFAULT_PARAMS,
    sanitize_params,
    compute_rates,
    expand_dependencies,
)


def test_compute_rates_basic():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, results = compute_rates(params)
    assert results["fixed_total"] == params["labor_cost"] + params["sga_cost"]
    assert results["required_profit_total"] == (
        params["loan_repayment"] + params["tax_payment"] + params["future_business"]
    )
    expected_annual = (
        (
            params["fulltime_workers"]
            + 0.75 * params["part1_workers"]
            + params["part2_coefficient"] * params["part2_workers"]
        )
        * params["working_days"]
        * (params["daily_hours"] * 60 * params["operation_rate"])
    )
    assert results["annual_minutes"] == expected_annual
    assert results["break_even_rate"] == results["fixed_total"] / results["annual_minutes"]
    assert results["required_rate"] == (
        results["fixed_total"] + results["required_profit_total"]
    ) / results["annual_minutes"]


def test_dependencies_and_no_cycle():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, _ = compute_rates(params)
    assert set(nodes["break_even_rate"]["depends_on"]) == {"fixed_total", "annual_minutes"}
    assert set(nodes["required_rate"]["depends_on"]) == {
        "fixed_total",
        "required_profit_total",
        "annual_minutes",
    }
    for key, node in nodes.items():
        assert key not in node["depends_on"]


def test_transitive_dependencies_link_to_rates():
    params = DEFAULT_PARAMS.copy()
    params, _ = sanitize_params(params)
    nodes, _ = compute_rates(params)
    deps = expand_dependencies(nodes)
    assert "labor_cost" in deps["break_even_rate"]
    assert "labor_cost" in deps["required_rate"]
    assert "loan_repayment" in deps["required_rate"]


def test_sanitize_params_negative():
    raw = DEFAULT_PARAMS.copy()
    raw.update(
        {
            "labor_cost": -100,
            "working_days": 0,
            "daily_hours": 0,
            "operation_rate": 0,
            "fulltime_workers": 0,
            "part1_workers": 0,
            "part2_workers": 0,
        }
    )
    params, warnings = sanitize_params(raw)
    assert params["labor_cost"] == 0
    assert params["working_days"] == 1
    assert params["daily_hours"] == 1
    assert params["operation_rate"] == 0.01
    assert params["fulltime_workers"] == 1.0
    assert warnings
