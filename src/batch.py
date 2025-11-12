#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for sweeping parameters and exporting CSV.

Parameters (via CLI args or env vars):
- --floors: comma list or range (e.g., 1-3) for number of floors (a)
- --layouts: subset of {T,L} (b)
- --occ: occupants per room list or range (e.g., 1-10) (c)
- --resp: responder counts list or range (e.g., 1-4) (n)
- --max-exit-combos: cap on exit assignments per n (m cap); default is all 2^n
- --out: output CSV path (default: OUT_DIR/batch_results.csv or ./output/batch_results.csv)

Usage examples:
  python src/batch.py --floors 1-2 --layouts T,L --occ 3-5 --resp 1-3 --max-exit-combos 8
  OUT_DIR=output LAYOUT_FILE=layout/layout_T.json python src/batch.py --floors 1 --occ 5 --resp 2
"""

import os
import sys
import csv
import json
import itertools as it
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(__file__))

from ilp_sweep import solve_building_sweep


def _parse_list_or_range(s: str) -> List[int]:
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x]
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _load_simple_layout(path: str):
    with open(path, "r", encoding="utf-8") as f:
        J = json.load(f)
    J_layout = J.get("layout") if isinstance(J.get("layout"), dict) else J

    # Try grid schema (doors)
    if "doors" in J_layout:
        xs = list(J_layout.get("doors", {}).get("xs", []))
        topZ = float(J_layout.get("doors", {}).get("topZ"))
        bottomZ = float(J_layout.get("doors", {}).get("bottomZ"))
        rooms = []
        for i, x in enumerate(xs):
            rooms.append((f"R{i}", (float(x), topZ)))
        n_top = len(xs)
        for i, x in enumerate(xs):
            rooms.append((f"R{n_top+i}", (float(x), bottomZ)))
        # responders fallback at frame ends
        frame = J_layout.get("frame", {})
        x1 = float(frame.get("x1", min(xs)))
        x2 = float(frame.get("x2", max(xs)))
        exits = {
            "E_L": (x1, (topZ + bottomZ) / 2.0),
            "E_R": (x2, (topZ + bottomZ) / 2.0),
        }
        return rooms, exits

    # Simple schema with 2D entries
    def _coord_from(v):
        if isinstance(v, dict):
            return (float(v.get("x", 0.0)), float(v.get("z", 0.0)))
        return (float(v), 0.0)

    rooms_raw = list(J_layout.get("rooms", []))
    rooms = [(r.get("id"), _coord_from(r.get("entry", r.get("pos", 0.0)))) for r in rooms_raw]
    responders_raw = J.get("responders", [])
    if len(responders_raw) >= 2:
        exits = {
            f"E_{responders_raw[0].get('id','F1')}": _coord_from(responders_raw[0].get("pos", 0.0)),
            f"E_{responders_raw[1].get('id','F2')}": _coord_from(responders_raw[1].get("pos", 0.0)),
        }
        # Normalize to E_L/E_R
        exits = {"E_L": list(exits.values())[0], "E_R": list(exits.values())[1]}
    else:
        xs = [p[0] for (_, (p0, _)) in enumerate(rooms)]
        exits = {"E_L": (min(xs) - 5.0, 0.0), "E_R": (max(xs) + 5.0, 0.0)}
    return rooms, exits


def _build_multi_floor(rooms_xy: List[Tuple[str, Tuple[float, float]]], exits_xy: Dict[str, Tuple[float, float]], floors: int, per_room_occ: int, floor_spacing: float = 60.0):
    rooms = []
    coords = {}
    occ = {}
    for f in range(floors):
        z_off = f * floor_spacing
        for (rid, (x, z)) in rooms_xy:
            nid = f"{rid}_F{f+1}"
            rooms.append(nid)
            coords[nid] = (x, z + z_off)
            occ[nid] = per_room_occ
    # exits (single pair for whole building)
    for k, (x, z) in exits_xy.items():
        coords[k] = (x, z)  # keep base level
    return rooms, coords, occ


def _all_exit_combos(n: int, cap=None):
    combos = []
    for bits in it.product(["L", "R"], repeat=n):
        combos.append(bits)
        if cap is not None and len(combos) >= cap:
            break
    return combos


def main():
    # Inputs
    floors_arg = os.environ.get("FLOORS", "1")
    layouts_arg = os.environ.get("LAYOUTS", "T,L")
    occ_arg = os.environ.get("OCC", "5")
    resp_arg = os.environ.get("RESP", "2")
    combo_cap = os.environ.get("MAX_EXIT_COMBOS")
    out_dir = os.environ.get("OUT_DIR", "output")
    out_csv = os.environ.get("OUT_CSV", os.path.join(out_dir, "batch_results.csv"))

    # CLI override
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default=floors_arg)
    ap.add_argument("--layouts", default=layouts_arg)
    ap.add_argument("--occ", default=occ_arg)
    ap.add_argument("--resp", default=resp_arg)
    ap.add_argument("--max-exit-combos", dest="max_exit", default=combo_cap)
    ap.add_argument("--out", default=out_csv)
    args = ap.parse_args()

    floors_list = _parse_list_or_range(args.floors)
    layouts_list = [s.strip().upper() for s in args.layouts.split(",") if s.strip()]
    occ_list = _parse_list_or_range(args.occ)
    resp_list = _parse_list_or_range(args.resp)
    max_exit = int(args.max_exit) if args.max_exit else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Pull SIM_CONFIG for solver parameters
    try:
        from configs import SIM_CONFIG as _SIM
    except Exception:
        _SIM = {}

    layout_paths = {"T": os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", "layout_T.json"),
                    "L": os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", "layout_L.json")}

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["floors", "layout", "per_room_occ", "responders", "exit_combo_id", "exit_combo", "makespan_s", "makespan_hms"]) 

        for floors in floors_list:
            for layout_code in layouts_list:
                layout_file = layout_paths.get(layout_code)
                if not layout_file or not os.path.exists(layout_file):
                    print(f"[skip] layout {layout_code} not found at {layout_file}")
                    continue
                rooms_xy, exits_xy = _load_simple_layout(layout_file)
                for occ in occ_list:
                    for n in resp_list:
                        # prepare responders
                        responders = [f"F{i+1}" for i in range(n)]
                        # build geometry for floors
                        rooms, coords, occupants = _build_multi_floor(rooms_xy, exits_xy, floors=floors, per_room_occ=occ)
                        # all exit combos
                        combos = _all_exit_combos(n, cap=max_exit)
                        for combo_id, combo in enumerate(combos):
                            start_node = {}
                            for i, side in enumerate(combo):
                                start_node[responders[i]] = "E_L" if side == "L" else "E_R"

                            result = solve_building_sweep(
                                rooms=rooms,
                                responders=responders,
                                start_node=start_node,
                                coords=coords,
                                occupants=occupants,
                                base_check_time=_SIM.get("base_check_time", 5.0),
                                time_per_occupant=_SIM.get("time_per_occupant", 3.0),
                                walk_speed=_SIM.get("walk_speed", 1.0),
                                verbose=False,
                                empirical_mode=_SIM.get("empirical_mode", "guide_low"),
                                redundancy_mode=_SIM.get("redundancy_mode", "per_responder_all_rooms"),
                                occupant_speed_high=_SIM.get("occupant_speed_high", 0.5),
                                occupant_speed_low=_SIM.get("occupant_speed_low", 0.3),
                                responder_speed_search=_SIM.get("responder_speed_search", 0.5),
                                responder_speed_carry=_SIM.get("responder_speed_carry", 0.25),
                                carry_capacity=_SIM.get("carry_capacity", 3),
                                comm_success=_SIM.get("comm_success", 0.85),
                                egress_distance_factor=_SIM.get("egress_distance_factor", 1.0),
                            )

                            # format time
                            def _fmt(seconds: float) -> str:
                                s = int(round(seconds or 0))
                                h = s // 3600; m = (s % 3600) // 60; sec = s % 60
                                return f"{h:02d}:{m:02d}:{sec:02d}"

                            makespan = float(result.get("T_total") or 0.0)
                            writer.writerow([
                                floors, layout_code, occ, n, combo_id,
                                "".join(combo),
                                makespan, _fmt(makespan)
                            ])
                        # flush per n
                        f.flush()

    print(f"Saved CSV: {args.out}")


if __name__ == "__main__":
    main()
