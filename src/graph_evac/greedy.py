"""Greedy approximations for the sweep planning problem."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .problem import EvacuationProblem

Assignment = Dict[str, List[str]]
Path = List[Tuple[str, str]]


def _is_exit(name: str) -> bool:
    return isinstance(name, str) and name.startswith("E_")


def _floor_of(name: str) -> int:
    if isinstance(name, str) and "_F" in name:
        try:
            return int(name.rsplit("_F", 1)[1])
        except Exception:
            return 1
    return 1


def _euclid(problem: EvacuationProblem, a: str, b: str) -> float:
    ax, ay = problem.node_coord(a)
    bx, by = problem.node_coord(b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def _nearest_exit_on_floor(problem: EvacuationProblem, node: str, floor: int, exits: List[str]) -> tuple[str | None, float]:
    best: str | None = None
    best_d = float("inf")
    for e in exits:
        if _floor_of(e) != floor:
            continue
        d = _euclid(problem, node, e)
        if d < best_d:
            best = e
            best_d = d
    if best is None:
        for e in exits:
            d = _euclid(problem, node, e)
            if d < best_d:
                best = e
                best_d = d
    return best, best_d


def _route_nodes(problem: EvacuationProblem, u: str, v: str) -> List[str]:
    if u == v:
        return [u]
    exits = list(problem.exits)
    if not exits:
        return [u, v]

    fu = _floor_of(u)
    fv = _floor_of(v)
    # 同一楼层，或者已经没有楼层概念时，直接连线
    if fu == fv:
        return [u, v]

    # 不同楼层：强制经由“本楼最近出口 → 目标楼最近出口”中转
    ex_u, _ = _nearest_exit_on_floor(problem, u, fu, exits)
    if _is_exit(v) and _floor_of(v) == fv:
        ex_v = v
    else:
        ex_v, _ = _nearest_exit_on_floor(problem, v, fv, exits)
    if ex_u is None or ex_v is None:
        return [u, v]
    if ex_u == ex_v:
        return [u, ex_u, v]
    return [u, ex_u, ex_v, v]


def _route_distance(problem: EvacuationProblem, u: str, v: str) -> float:
    nodes = _route_nodes(problem, u, v)
    if len(nodes) <= 1:
        return 0.0
    total = 0.0
    for a, b in zip(nodes, nodes[1:]):
        total += _euclid(problem, a, b)
    return total


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
    load = defaultdict(float)
    for room_id in problem.room_ids:
        best_rid = min(
            problem.responder_ids,
            key=lambda rid: (load[rid], _route_distance(problem, problem.responders[rid].start_node, room_id)),
        )
        assignments[best_rid].append(room_id)
        load[best_rid] += _route_distance(problem, problem.responders[best_rid].start_node, room_id)
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
        next_room = min(unvisited, key=lambda room: _route_distance(problem, curr, room))
        unvisited.remove(next_room)
        return next_room

    while True:
        nxt = _pop_next(current)
        if nxt is None:
            break
        segment_nodes = _route_nodes(problem, current, nxt)
        for a, b in zip(segment_nodes, segment_nodes[1:]):
            travel = _euclid(problem, a, b) / max(problem.config.walk_speed, 1e-6)
            total_time += travel
            path.append((a, b))
        total_time += problem.room_service_time(nxt)
        current = nxt

    if current != start_node:
        exits = list(problem.exits)
        if not exits:
            exits = [problem.responders[rid].start_node for rid in problem.responder_ids]
        first_floor_exits = [e for e in exits if _floor_of(e) == 1]
        target_exits = first_floor_exits or exits
        nearest_exit = min(target_exits, key=lambda e: _route_distance(problem, current, e))
        segment_nodes = _route_nodes(problem, current, nearest_exit)
        for a, b in zip(segment_nodes, segment_nodes[1:]):
            travel = _euclid(problem, a, b) / max(problem.config.walk_speed, 1e-6)
            total_time += travel
            path.append((a, b))

    return path, total_time
