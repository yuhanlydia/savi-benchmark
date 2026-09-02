#!/usr/bin/env bash
set -euo pipefail

# Wait for a specific Experiment 0B process, then run every post-collection
# artifact producer exactly once.  This script never interprets partial data as
# a confirmatory result and never overwrites an existing judge-labelled file.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${EXP0B_CONFIG:-configs/exp0b_math.yaml}"
pid="${1:-}"
cd "$repo_root"

if [[ -n "$pid" ]]; then
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
fi

status_json="$(.venv/bin/python -m savi.monitor --config "$config")"
read -r completed planned < <(.venv/bin/python -c \
  'import json,sys; x=json.load(sys.stdin); print(x["completed_rows"],x["planned_jobs"])' <<<"$status_json")
if [[ "$completed" != "$planned" ]]; then
  echo "Experiment 0B ended before completion: $completed/$planned" >&2
  exit 2
fi

root="$(.venv/bin/python -c \
  'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output"]["root"])' "$config" 2>/dev/null || true)"
if [[ -z "$root" ]]; then
  root="outputs/exp0b_math"
fi

.venv/bin/python -m savi.analysis --config "$config"
.venv/bin/python -m savi.analysis --config "$config" --nonterminal-only
.venv/bin/python -m savi.oracle_analysis --config "$config" \
  --values "$root/state_values.jsonl" --output "$root/oracle_report.json"
.venv/bin/python -m savi.oracle_analysis --config "$config" \
  --values "$root/state_values_nonterminal.jsonl" \
  --output "$root/oracle_report_nonterminal.json"
.venv/bin/python -m savi.prefix_audit --config "$config" \
  --output "$root/prefix_audit.json"
.venv/bin/python -m savi.judge_handoff --config "$config" export \
  --output "$root/judge_queue.jsonl"

echo "Experiment 0B postprocessing complete: $root"
