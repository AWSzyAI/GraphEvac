#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy sweep planner (no ILP):
- Assign each room to the nearest responder start.
- For each responder, visit assigned rooms via nearest-neighbour order.
- After last room, return to the nearest exit (denoted as G_{kid}).
- First-clear times are computed from simple walk times + base_check_time.

Notes
-----
- Distances are straight-line over provided coords. For cross-floor moves
  between two non-exit nodes, we approximate travel by routing via the
  nearest exit on the source floor to the nearest exit on the target floor.
- Exits are inferred from `coords` keys starting with "E_"; fallback to
  `start_node.values()` if none found.
"""

from typing import Dict, List, Tuple
import math


def _is_exit(name: str) -> bool:
    return isinstance(name, str) and name.startswith("E_")


def _floor_of(name: str) -> int:
    if isinstance(name, str) and "_F" in name:
        try:
            return int(name.rsplit("_F", 1)[1])
        except Exception:
            return 1
    return 1


def _euclid(a, b) -> float:
    if isinstance(a, (tuple, list)):
        dx = float(a[0]) - float(b[0])
        dz = float(a[1]) - float(b[1])
        return (dx * dx + dz * dz) ** 0.5
    return abs(float(a) - float(b))


def _nearest_exit_on_floor(node_name: str, floor: int, coords: Dict[str, Tuple[float, float]], exit_nodes: List[str]):
    c0 = coords[node_name]
    best = None
    best_d = float("inf")
    # prefer exits explicitly on the same floor if available
    for e in exit_nodes:
        if _floor_of(e) != floor:
            continue
        d = _euclid(c0, coords[e])
        if d < best_d:
            best = e
            best_d = d
    # fallback to any exit if same-floor not present
    if best is None:
        for e in exit_nodes:
            d = _euclid(c0, coords[e])
            if d < best_d:
                best = e
                best_d = d
    return best, best_d


def _dist_with_exits(u: str, v: str, coords: Dict[str, Tuple[float, float]], exit_nodes: List[str]) -> float:
    if u == v:
        return 0.0
    # direct if either end is an exit or on same floor
    if _is_exit(u) or _is_exit(v) or _floor_of(u) == _floor_of(v):
        return _euclid(coords[u], coords[v])
    # route via exits: u -> exit(floor u) -> exit(floor v) -> v
    fu = _floor_of(u)
    fv = _floor_of(v)
    ex_u, d1 = _nearest_exit_on_floor(u, fu, coords, exit_nodes)
    ex_v, d2 = _nearest_exit_on_floor(v, fv, coords, exit_nodes)
    if ex_u is None or ex_v is None:
        # fallback: direct euclid (should be rare)
        return _euclid(coords[u], coords[v])
    d_mid = _euclid(coords[ex_u], coords[ex_v])
    return d1 + d_mid + d2


def solve_building_sweep(
    rooms: List[str],
    responders: List[str],
    start_node: Dict[str, str],
    coords: Dict[str, Tuple[float, float]],
    occupants: Dict[str, int],
    base_check_time: float = 5.0,
    time_per_occupant: float = 3.0,
    walk_speed: float = 1.0,
    *,
    empirical_mode=None,
    occupant_speed_high: float = 0.5,    # reserved (not used in greedy)
    occupant_speed_low: float = 0.3,     # reserved (not used in greedy)
    responder_speed_search: float = 0.5, # reserved (not used in greedy)
    responder_speed_carry: float = 0.25,
    carry_capacity: int = 3,
    comm_success: float = 0.85,          # reserved (not used in greedy)
    redundancy_mode: str = "assignment", # kept for meta
    egress_distance_factor: float = 1.0,
    solver=None,                         # compatibility only
    verbose: bool = False,               # compatibility only
):
    # infer exits
    exit_nodes = [k for k in coords.keys() if _is_exit(k)]
    if not exit_nodes:
        exit_nodes = list(dict.fromkeys(start_node.values()))

    # per-room sweep dwell (now depends on occupants)
    sweep_time = {
        r: float(base_check_time) + float(time_per_occupant) * float(max(0, int(occupants.get(r, 0))))
        for r in rooms
    }

    # assign each room to nearest responder start (travel via exits for cross-floor)
    # tie-break: balance load among responders with equal distance (keeps runtime O(N*M))
    assignment: Dict[str, str] = {}
    assigned_load: Dict[str, float] = {kid: 0.0 for kid in responders}
    EPS = 1e-6
    for r in rooms:
        # distances from all starts
        dists = {kid: _dist_with_exits(start_node[kid], r, coords, exit_nodes) for kid in responders}
        dmin = min(dists.values()) if dists else float("inf")
        cand = [kid for kid, d in dists.items() if abs(d - dmin) <= EPS]
        if len(cand) == 1:
            chosen = cand[0]
        else:
            # balance by current load (approximate) among ties
            chosen = min(cand, key=lambda kid: assigned_load.get(kid, 0.0))
        assignment[r] = chosen
        assigned_load[chosen] = assigned_load.get(chosen, 0.0) + sweep_time[r]

    # build greedy route per responder
    responders_info: Dict[str, dict] = {}
    for kid in responders:
        s = start_node[kid]
        assigned_rooms = [r for r in rooms if assignment[r] == kid]
        rem = set(assigned_rooms)
        path = []  # list of (u, v)
        cur = s
        # nearest-neighbour heuristic
        while rem:
            nxt = min(rem, key=lambda r: _dist_with_exits(cur, r, coords, exit_nodes))
            path.append((cur, nxt))
            cur = nxt
            rem.remove(nxt)
        # return to nearest exit via dummy G_k
        if cur != s:
            path.append((cur, f"G_{kid}"))

        # compute time breakdown (walk + dwell + optional carry/egress)
        t = 0.0
        first_clear = {}
        for i, (u, v) in enumerate(path):
            # walking to next target
            if v.startswith("G_"):
                if str(empirical_mode).lower() == "carry":
                    # In carry mode, the last room's final egress trip time is accounted below
                    d_walk = 0.0
                else:
                    ex, droom = _nearest_exit_on_floor(u, _floor_of(u), coords, exit_nodes)
                    d_walk = droom
            else:
                d_walk = _dist_with_exits(u, v, coords, exit_nodes)
            t += d_walk / max(float(walk_speed), 1e-6)

            if v in rooms:
                # dwell/clear time
                t += sweep_time.get(v, 0.0)
                first_clear[v] = {"time": float(t), "by": kid}

                # optional carry/egress overhead (lightweight approx.)
                if str(empirical_mode).lower() == "carry":
                    occ_v = max(0, int(occupants.get(v, 0)))
                    if occ_v > 0:
                        trips = math.ceil(occ_v / max(int(carry_capacity) if carry_capacity else 1, 1))
                        ex, droom = _nearest_exit_on_floor(v, _floor_of(v), coords, exit_nodes)
                        egress_d = float(droom) * float(egress_distance_factor)
                        # if last room (next edge goes to G_k), only need go-to-exit once (no return)
                        is_last_room = (i == len(path) - 2) and path[-1][1].startswith("G_")
                        if is_last_room:
                            carry_time = trips * (egress_d / max(float(responder_speed_carry), 1e-6))
                        else:
                            carry_time = trips * (2.0 * egress_d / max(float(responder_speed_carry), 1e-6))
                        t += carry_time

        responders_info[kid] = {
            "T": float(t),
            "assigned_rooms": assigned_rooms,
            "path": path,
        }

    # global first-clear aggregation
    by_room = {}
    for kid, info in responders_info.items():
        # recompute first-clear from path for global view
        tcur = 0.0
        for (u, v) in info.get("path", []):
            if v.startswith("G_"):
                ex, droom = _nearest_exit_on_floor(u, _floor_of(u), coords, exit_nodes)
                d_walk = droom
            else:
                d_walk = _dist_with_exits(u, v, coords, exit_nodes)
            tcur += d_walk / max(float(walk_speed), 1e-6)
            if v in rooms:
                tcur += sweep_time.get(v, 0.0)
                # first arrival defines first-clear
                if v not in by_room or tcur < by_room[v]["time"]:
                    by_room[v] = {"time": float(tcur), "by": kid}

    order = sorted(by_room.keys(), key=lambda r: by_room[r]["time"]) if by_room else []

    result = {
        "status": "GREEDY",
        "T_total": max([info["T"] for info in responders_info.values()] or [0.0]),
        "responders": responders_info,
        "first_clear": {"by_room": by_room, "order": order},
        "sweep_time": sweep_time,
        "meta": {
            "walk_speed_used": float(walk_speed),
            "redundancy_mode": "greedy",
            "exits": list(exit_nodes),
            "occupant_time_model": {
                "base_check_time": float(base_check_time),
                "time_per_occupant": float(time_per_occupant),
                "carry_mode": str(empirical_mode).lower() == "carry",
                "carry_capacity": int(max(carry_capacity, 1)),
                "responder_speed_carry": float(responder_speed_carry),
                "egress_distance_factor": float(egress_distance_factor),
            },
        },
    }

    return result

