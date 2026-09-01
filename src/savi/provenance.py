from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

from .io import write_json


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(config_path: str | Path) -> dict:
    import torch
    import transformers

    return {
        "created_unix": time.time(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "code_commit": _command("git", "rev-parse", "HEAD"),
        "code_dirty": bool(_command("git", "status", "--porcelain")),
        "r3bench_commit": _command("git", "-C", "third_party/R-3-Bench", "rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": _command(
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ),
    }


def write_once(config_path: str | Path, output_path: str | Path) -> dict:
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite provenance: {target}")
    value = collect(config_path)
    write_json(target, value)
    return value
