import json
import os
from pathlib import Path

import pytest


@pytest.mark.regression
def test_regression_cases_smoke():
    """
    Heavy regression tests are opt-in because they may hit network/cache and take minutes.
    Enable by setting RUN_REGRESSION=1.
    """
    if os.getenv("RUN_REGRESSION", "0") != "1":
        pytest.skip("RUN_REGRESSION!=1 (opt-in)")

    from regression.runner import run_regression_case

    cases_dir = Path(__file__).resolve().parents[1] / "regression" / "cases"
    case_files = sorted(cases_dir.glob("*.json"))
    assert case_files, "No regression cases found"

    failures = []
    for cf in case_files:
        res = run_regression_case(str(cf), artifacts_root="output/regression")
        if not res.ok:
            failures.append((cf.name, res.report.get("assertions")))

    assert not failures, f"Regression failures: {json.dumps(failures, ensure_ascii=False, indent=2)}"


