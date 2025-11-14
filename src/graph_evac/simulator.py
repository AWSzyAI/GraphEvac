"""Basic timeline simulator used for visualisation outputs."""
from __future__ import annotations

from typing import Dict, List

from .problem import EvacuationProblem
from .config import Config


def simulate_sweep(problem: EvacuationProblem, plan: dict, config: Config) -> Dict[str, List[dict]]:
    timelines: Dict[str, List[dict]] = {}
    for rid, info in plan.get("responders", {}).items():
        timeline: List[dict] = []
        current_time = 0.0
        current = problem.responders[rid].start_node
        for edge in info.get("path", []):
            start, end = edge
            travel_time = problem.distance(start, end) / max(config.walk_speed, 1e-6)
            timeline.append(
                {
                    "type": "walk",
                    "from": start,
                    "to": end,
                    "t0": current_time,
                    "t1": current_time + travel_time,
                }
            )
            current_time += travel_time
            if end in problem.rooms:
                dwell = problem.room_service_time(end)
                timeline.append(
                    {
                        "type": "inspect",
                        "room": end,
                        "t0": current_time,
                        "t1": current_time + dwell,
                    }
                )
                current_time += dwell
            current = end
        timelines[rid] = timeline
    return timelines
