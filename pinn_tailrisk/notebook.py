"""Small helpers for notebooks: show a stage's figures and load its metrics."""
from __future__ import annotations

import json

from .config import RESULTS
from .plotting import FIG_DIR


def show_figures(stage: str, names: list[str] | None = None, width: int = 950) -> None:
    """Display figures/<stage>/*.png inline (all of them, or the given names, in order)."""
    from IPython.display import Image, Markdown, display
    paths = sorted((FIG_DIR / stage).glob("*.png")) if names is None else [FIG_DIR / stage / f"{n}.png" for n in names]
    if not paths:
        display(Markdown(f"_No figures in `figures/{stage}` yet; run the stage first._"))
    for p in paths:
        display(Markdown(f"**{p.stem}**"))
        display(Image(filename=str(p), width=width))


def load_metrics(stage: str, name: str = "metrics") -> dict:
    return json.loads((RESULTS / stage / f"{name}.json").read_text())
