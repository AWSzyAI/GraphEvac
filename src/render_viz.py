#!/usr/bin/env python3
"""CLI helper to rebuild visualization assets from saved runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from graph_evac import Config, EvacuationProblem, expand_floors, load_layout
from graph_evac.visualizer import export_visuals


def _resolve_layout_path(layout_file: str) -> Path:
    path = Path(layout_file)
    if path.exists():
        return path
    repo_root = CURRENT_DIR.parent
    candidate = (repo_root / layout_file).resolve()
    if candidate.exists():
        return candidate
    return path


def _find_latest_run(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    candidates = []
    for child in output_root.iterdir():
        plan_path = child / "plan.json"
        if plan_path.exists():
            candidates.append((plan_path.stat().st_mtime, child))
    if not candidates:
        return None
    _, latest = max(candidates, key=lambda item: item[0])
    return latest


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    default_root = os.environ.get("OUTPUT_ROOT", "output")
    parser = argparse.ArgumentParser(description="Rebuild GraphEvac visualization outputs")
    parser.add_argument("--run-dir", help="Specific run directory containing plan.json", default=None)
    parser.add_argument("--output-root", default=default_root, help="Root directory storing run outputs")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _find_latest_run(output_root)
    if not run_dir:
        raise SystemExit("No prior run artifacts found. Run `make run` first or provide --run-dir.")

    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"plan.json not found under {run_dir}")
    payload = _load_json(plan_path)
    plan_data = payload.get("plan", {})
    config_data = payload.get("config", {})
    config = Config(**config_data)

    timeline_path = run_dir / "timeline.json"
    timeline = _load_json(timeline_path) if timeline_path.exists() else {}

    layout_path = _resolve_layout_path(config.layout_file)
    layout = load_layout(layout_path)
    layout, exits = expand_floors(layout, config.floors, config.floor_spacing)
    problem = EvacuationProblem.from_layout(layout, config, exits)

    artifacts = export_visuals(problem, config, plan_data, timeline, run_dir)
    if artifacts:
        names = ", ".join(f"{key}:{path.name}" for key, path in artifacts.items())
        print(f"Rebuilt visualization assets ({names}) in {run_dir}")
    else:
        print("No plan data found to visualize.")


if __name__ == "__main__":
    main()

