"""Visualization export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import Config
from .problem import EvacuationProblem
from .reporting import build_viz_payload

import viz


def export_visuals(
    problem: EvacuationProblem,
    config: Config,
    plan_data: dict,
    timeline: Dict[str, list] | None,
    output_dir: Path,
    payload: dict | None = None,
) -> Dict[str, Path]:
    """Render path heatmap, Gantt chart, and GIF animation for a run."""

    payload = payload or build_viz_payload(problem, plan_data, timeline, config)
    if not payload:
        return {}

    start_nodes = {rid: resp.start_node for rid, resp in problem.responders.items()}
    coords = problem.coords
    rooms = list(problem.room_ids)

    artifacts: Dict[str, Path] = {}

    heatmap_path = output_dir / "heatmap.png"
    viz.plot_paths(coords, rooms, start_nodes, payload, savepath=str(heatmap_path))
    artifacts["heatmap"] = heatmap_path

    gantt_path = output_dir / "gantt.png"
    viz.plot_gantt(coords, rooms, start_nodes, payload, savepath=str(gantt_path))
    artifacts["gantt"] = gantt_path

    gif_path = output_dir / "anim.gif"
    viz.animate_gif(coords, rooms, start_nodes, payload, savepath=str(gif_path))
    artifacts["gif"] = gif_path

    return artifacts
