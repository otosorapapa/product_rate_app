import sys
from pathlib import Path

import pandas as pd

# ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import compute_results


def test_compute_results_handles_duplicate_columns():
    df = pd.DataFrame(
        {
            "product_no": [1],
            "product_name": ["A"],
            "actual_unit_price": [100],
            "material_unit_cost": [40],
            "subcontract_cost": [10],
            "daily_qty": [5],
            "minutes_per_unit": [2],
        }
    )
    # introduce duplicate column that previously caused ValueError
    df["daily_total_minutes"] = 10
    df = pd.concat([df, df[["daily_total_minutes"]]], axis=1)

    result = compute_results(df, 10, 15)
    assert result.loc[0, "va_per_min"] == 25.0
