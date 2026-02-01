from __future__ import annotations

import argparse
import sys

from regression.runner import run_regression_case


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a single regression generation case and save artifacts + report.")
    p.add_argument("--case", required=True, help="Path to case JSON (e.g. regression/cases/paton_bridge.json)")
    p.add_argument("--artifacts", default="output/regression", help="Artifacts output root directory")
    args = p.parse_args(argv)

    res = run_regression_case(case_path=args.case, artifacts_root=args.artifacts)
    print(f"[REGRESSION] case={res.case_name} ok={res.ok} artifacts={res.artifacts_dir}")
    if not res.ok:
        print("[REGRESSION] FAILED assertions:")
        for k, v in (res.report.get("assertions") or {}).items():
            if isinstance(v, dict) and v.get("ok") is False:
                print(f"  - {k}: {v}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


