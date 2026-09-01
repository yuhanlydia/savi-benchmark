#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

command -v python3 >/dev/null
command -v gcc >/dev/null || {
  echo "A C compiler is required by the installed PyTorch/Triton runtime." >&2
  exit 1
}

git submodule update --init --recursive
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e third_party/R-3-Bench -e . pytest

model_revision="$([ -f resources.lock.json ] && .venv/bin/python - <<'PY'
import json
print(json.load(open("resources.lock.json"))["model"]["commit"])
PY
)"
.venv/bin/hf download Qwen/Qwen3-8B --revision "$model_revision" --local-dir models/Qwen3-8B
.venv/bin/python -m pytest -q

echo "Bootstrap complete. Activate with: source .venv/bin/activate"
