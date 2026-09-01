from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import load_yaml, read_jsonl, write_json
from .monitor import build_health_report


def summarize(config):
    root = Path(config["output"]["root"])
    rows = read_jsonl(config["output"]["continuations"])
    telemetry_path = root / "telemetry.jsonl"
    telemetry = read_jsonl(telemetry_path) if telemetry_path.exists() else []
    provenance_path = root / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    disposition_path = root / "run_disposition.json"
    disposition = json.loads(disposition_path.read_text()) if disposition_path.exists() else None
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["spent_budget"]), int(row["horizon"]))].append(row)
    conditions = []
    for (spent, horizon), items in sorted(groups.items()):
        conditions.append({
            "spent_budget": spent, "horizon": horizon, "records": len(items),
            "parseable_fraction": float(np.mean([bool(row["parsed_answer_normalized"]) for row in items])),
            "exact_correct_fraction": float(np.mean([bool(row["correct_exact_normalized"]) for row in items])),
        })
    gpu_rows = [row["gpu"] for row in telemetry if row.get("gpu")]
    gpu = None
    if gpu_rows:
        gpu = {
            "snapshots": len(gpu_rows),
            "mean_temperature_c": float(np.mean([row["gpu_temperature_c"] for row in gpu_rows])),
            "max_temperature_c": float(np.max([row["gpu_temperature_c"] for row in gpu_rows])),
            "mean_utilization_pct": float(np.mean([row["gpu_utilization_pct"] for row in gpu_rows])),
            "max_memory_mib": float(np.max([row["gpu_memory_mib"] for row in gpu_rows])),
        }
    elapsed = None
    records_per_hour = None
    if provenance and rows:
        end = max([row.get("timestamp_unix", 0) for row in telemetry] +
                  [Path(config["output"]["continuations"]).stat().st_mtime])
        elapsed = max(0.0, end - float(provenance["created_unix"]))
        records_per_hour = len(rows) / elapsed * 3600 if elapsed else None
    return {
        "experiment": config["experiment"]["name"],
        "health": build_health_report(config), "conditions": conditions,
        "elapsed_seconds_observed": elapsed, "records_per_hour": records_per_hour,
        "gpu": gpu, "provenance": provenance, "disposition": disposition,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--output")
    args = parser.parse_args(); config = load_yaml(args.config); report = summarize(config)
    if args.output: write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
