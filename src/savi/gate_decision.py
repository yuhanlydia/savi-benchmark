from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide_gap_gates(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    mode = str(config.get("gates", {}).get("primary_aliasing_gate", "raw_q"))
    if mode == "marginal_value":
        ci = report.get("noise_corrected_gain_variance_problem_bootstrap_ci95") or [None, None]
        variance_positive = ci[0] is not None and float(ci[0]) > 0.0
        aliasing_pass = bool(report.get("mva_gate_pass")) and variance_positive
    elif mode == "raw_q":
        variance_positive = None
        aliasing_pass = bool(report.get("gate_0_pass"))
    else:
        raise ValueError(f"Unknown gates.primary_aliasing_gate={mode!r}")
    decision_pass = bool(report.get("gate_1_pass"))
    return {
        "primary_aliasing_gate": mode,
        "aliasing_gate_pass": aliasing_pass,
        "noise_corrected_variance_ci_lower_positive": variance_positive,
        "decision_relevance_gate_pass": decision_pass,
        "overall_gap_gate_pass": aliasing_pass and decision_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    from .io import load_yaml

    config = load_yaml(args.config)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    decision = decide_gap_gates(config, report)
    text = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
