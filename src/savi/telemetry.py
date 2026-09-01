from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from .io import append_jsonl, load_yaml
from .monitor import build_health_report


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True).strip().splitlines()[0]
        values = [value.strip() for value in line.split(",")]
        return dict(zip(("gpu_utilization_pct", "gpu_memory_mib", "gpu_temperature_c",
                         "gpu_power_w", "gpu_sm_clock_mhz"), map(float, values)))
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def snapshot(config):
    return {"timestamp_unix": time.time(), "health": build_health_report(config),
            "gpu": gpu_snapshot()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--output")
    args = parser.parse_args(); config = load_yaml(args.config)
    row = snapshot(config)
    output = args.output or str(Path(config["output"]["root"]) / "telemetry.jsonl")
    append_jsonl(output, [row]); print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
