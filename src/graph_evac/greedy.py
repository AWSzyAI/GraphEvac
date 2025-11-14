"""Greedy approximations for the sweep planning problem."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .problem import EvacuationProblem

Assignment = Dict[str, List[str]]
Path = List[Tuple[str, str]]


def plan(problem: EvacuationProblem) -> dict:
    assignments = _assign_rooms(problem)
    responder_plans = {}
    makespan = 0.0
    for rid, rooms in assignments.items():
        edges, total_time = _build_path(problem, rid, rooms)
        responder_plans[rid] = {
            "path": edges,
            "rooms": list(rooms),
            "time": total_time,
        }
        makespan = max(makespan, total_time)
    return {
        "algorithm": "greedy",
        "responders": responder_plans,
        "makespan": makespan,
    }


def _assign_rooms(problem: EvacuationProblem) -> Assignment:
    mode = problem.config.redundancy_mode
    assignments: Assignment = {rid: [] for rid in problem.responder_ids}
    if mode == "per_responder_all_rooms":
        for rid in assignments:
            assignments[rid] = list(problem.room_ids)
        return assignments

    if mode == "double_check":
        for room_id in problem.room_ids:
            ordered = sorted(
                problem.responder_ids,
                key=lambda rid: problem.distance(problem.responders[rid].start_node, room_id),
            )
            chosen = ordered[:2] if len(ordered) >= 2 else ordered
            for rid in chosen:
                assignments[rid].append(room_id)
        return assignments

    # default "assignment" mode
    load = defaultdict(int)
    for room_id in problem.room_ids:
        best_rid = min(
            problem.responder_ids,
            key=lambda rid: (load[rid], problem.distance(problem.responders[rid].start_node, room_id)),
        )
        assignments[best_rid].append(room_id)
        load[best_rid] += 1
    return assignments


def _build_path(problem: EvacuationProblem, rid: str, rooms: Iterable[str]) -> tuple[Path, float]:
    responder = problem.responders[rid]
    start_node = responder.start_node
    unvisited = list(rooms)
    current = start_node
    path: Path = []
    total_time = 0.0

    def _pop_next(curr: str) -> str | None:
        if not unvisited:
            return None
        next_room = min(unvisited, key=lambda room: problem.distance(curr, room))
        unvisited.remove(next_room)
        return next_room

    while True:
        nxt = _pop_next(current)
        if nxt is None:
            break
        travel = problem.distance(current, nxt) / max(problem.config.walk_speed, 1e-6)
        total_time += travel
        path.append((current, nxt))
        total_time += problem.room_service_time(nxt)
        current = nxt

    if current != start_node:
        travel = problem.distance(current, start_node) / max(problem.config.walk_speed, 1e-6)
        total_time += travel
        path.append((current, start_node))

    return path, total_time
