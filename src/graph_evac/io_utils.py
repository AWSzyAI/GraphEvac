"""Helper utilities for saving planner output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _sanitize(segment: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in segment)


def prepare_output_dir(
    output_root: str,
    layout_file: str,
    responders: Iterable[str],
    redundancy_mode: str,
    floors: int,
    total_occupants: int,
) -> Path:
    layout_label = _sanitize(Path(layout_file).stem)
    resp_label = _sanitize("-".join(sorted(responders)))
    redundancy_label = _sanitize(redundancy_mode)
    floors_label = f"floors_{floors}"
    occ_label = f"occ_{total_occupants}"
    tag = f"layout_{layout_label}_{floors_label}_resp_{resp_label}_{occ_label}_red_{redundancy_label}"
    output_dir = Path(output_root) / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_plan(output_dir: Path, plan: Dict[str, Any]) -> Path:
    path = output_dir / "plan.json"
    save_json(path, plan)
    return path


def save_timeline(output_dir: Path, timeline: Dict[str, Any]) -> Path:
    path = output_dir / "timeline.json"
    save_json(path, timeline)
    return path
