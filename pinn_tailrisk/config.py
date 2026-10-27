"""Experiment configs (YAML) and result bookkeeping."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_config(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def save_results(stage: str, name: str, payload: dict) -> Path:
    """results/<stage>/<name>.json, stamped with git hash and machine info."""
    out = RESULTS / stage
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(git=git_hash(), python=platform.python_version(), torch=None, device="cpu",
                cpu=platform.processor())
    try:  # reference stages run without torch
        import torch
        meta.update(torch=torch.__version__,
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    path = out / f"{name}.json"
    path.write_text(json.dumps(dict(meta=meta, **payload), indent=2, default=float))
    return path
