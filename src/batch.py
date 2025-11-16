#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for sweeping parameters and exporting CSV.

Parameters (via CLI args or env vars):
- --floors: comma list or range (e.g., 1-3) for number of floors (a)
- --layouts: subset of {BASELINE,A,B} plus legacy aliases (b)
- --occ: occupants per room list or range (e.g., 1-10) (c)
- --resp: responder counts list or range (e.g., 1-4) (n)
- --max-exit-combos: cap on exit assignments per n (m cap); default is all 2^n
- --out: output CSV path (default: OUT_DIR/batch_results.csv or ./output/batch_results.csv)

Usage examples:
  python src/batch.py --floors 1-2 --layouts BASELINE,A,B --occ 3-5 --resp 1-3 --max-exit-combos 8
  OUT_DIR=output LAYOUT_FILE=layout/layout_T.json python src/batch.py --floors 1 --occ 5 --resp 2
"""

import os
import sys
import csv
import json
import itertools as it
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(__file__))

from configs import BATCH_CONFIG

from greedy_sweep import solve_building_sweep


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
        # rooms is a list of (room_id, (x,z))
        xs = [xy[0] for (_rid, xy) in rooms]
        exits = {"E_L": (min(xs) - 5.0, 0.0), "E_R": (max(xs) + 5.0, 0.0)}
    return rooms, exits


def _build_multi_floor(rooms_xy: List[Tuple[str, Tuple[float, float]]], exits_xy: Dict[str, Tuple[float, float]], floors: int, per_room_occ: int, floor_spacing: float = 20.0):
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


def _equal_split_combos(n: int):
    """Reduced combos: just left/right splits plus alternates."""
    if n <= 0:
        return []
    left_count = (n + 1) // 2
    right_count = n // 2

    combos = []
    left_major = tuple(["L"] * left_count + ["R"] * right_count)
    right_major = tuple(["R"] * left_count + ["L"] * right_count)
    combos.append(left_major)
    combos.append(right_major)

    if n % 2 == 0:
        alternating = tuple(["L" if i % 2 == 0 else "R" for i in range(n)])
        combos.append(alternating)
    else:
        combos.append(tuple(["L" if i % 2 == 0 else "R" for i in range(n)]))
        combos.append(tuple(["R" if i % 2 == 0 else "L" for i in range(n)]))

    seen = set()
    uniq = []
    for c in combos:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _format_responder_orders(result: dict, responders: List[str], rooms: List[str]) -> str:
    room_set = set(rooms)
    parts = []
    for rid in responders:
        info = result.get("responders", {}).get(rid, {})
        seq = [v for (_, v) in info.get("path", []) if v in room_set]
        parts.append(f"{rid}:{'->'.join(seq) if seq else 'N/A'}")
    return " | ".join(parts)


def _format_start_positions(start_node: Dict[str, str], responders: List[str]) -> str:
    return ";".join(f"{rid}:{start_node.get(rid, 'N/A')}" for rid in responders)


def main():
    # Inputs (env override or batch defaults defined in configs.py)
    floors_arg = os.environ.get("FLOORS", BATCH_CONFIG.get("floors", "2"))
    layouts_arg = os.environ.get("LAYOUTS", BATCH_CONFIG.get("layouts", "BASELINE,A,B"))
    occ_arg = os.environ.get("OCC", BATCH_CONFIG.get("occ", "5"))
    resp_arg = os.environ.get("RESP", BATCH_CONFIG.get("resp", "2"))
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
    ap.add_argument("--summary-out", dest="summary_out", default=os.path.join(out_dir, "summary_results.csv"))
    args = ap.parse_args()

    floors_list = _parse_list_or_range(args.floors)
    layouts_list = [s.strip().upper() for s in args.layouts.split(",") if s.strip()]
    # Only auto-append BASELINE when --layouts 未在命令行中显式提供
    provided_layouts_cli = any(a.startswith("--layouts") for a in sys.argv[1:])
    if (not provided_layouts_cli) and ("BASELINE" not in layouts_list and "BASE" not in layouts_list):
        layouts_list.append("BASELINE")
    occ_list = _parse_list_or_range(args.occ)
    resp_list = _parse_list_or_range(args.resp)
    max_exit = int(args.max_exit) if args.max_exit else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    select_out = os.environ.get("SELECT_OUT", os.path.join(out_dir, "select_result.csv"))

    # Pull SIM_CONFIG for solver parameters
    try:
        from configs import SIM_CONFIG as _SIM
    except Exception:
        _SIM = {}

    base_dir = os.path.dirname(os.path.dirname(__file__))
    layout_paths = {
        "A": os.path.join(base_dir, "layout", "layout_A.json"),
        "LAYOUT_A": os.path.join(base_dir, "layout", "layout_A.json"),
        "B": os.path.join(base_dir, "layout", "layout_B.json"),
        "LAYOUT_B": os.path.join(base_dir, "layout", "layout_B.json"),
        "T": os.path.join(base_dir, "layout", "layout_T.json"),
        "L": os.path.join(base_dir, "layout", "layout_L.json"),
        "BASE": os.path.join(base_dir, "layout", "baseline.json"),
        "BASELINE": os.path.join(base_dir, "layout", "baseline.json"),
    }

    # Track best per (layout, floors, responders)
    best_by_combo = {}
    # Track best per (layout, per_room_occ)
    best_by_layout_occ = {}

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layout",
            "floors",
            "per_room_occ",
            "responders",
            "exit_combo",
            "makespan_hms",
            "room_clear_order",
            "exit_combo_id",
            "makespan_s",
            "responder_orders",
            "start_positions",
        ])

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
                        # Reduce combos to equitable splits between two exits
                        combos = _equal_split_combos(n)
                        if max_exit:
                            combos = combos[:max(0, int(max_exit))]
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
                            room_order = result.get("first_clear", {}).get("order", [])
                            room_order_str = "->".join(room_order)
                            responder_orders = _format_responder_orders(result, responders, rooms)
                            start_positions = _format_start_positions(start_node, responders)
                            writer.writerow([
                                layout_code,
                                floors,
                                occ,
                                n,
                                "".join(combo),
                                _fmt(makespan),
                                room_order_str,
                                combo_id,
                                makespan,
                                responder_orders,
                                start_positions,
                            ])
                            key = (layout_code, floors, n)
                            cur_best_combo = best_by_combo.get(key)
                            if cur_best_combo is None or makespan < cur_best_combo["makespan_s"]:
                                best_by_combo[key] = {
                                    "layout": layout_code,
                                    "floors": floors,
                                    "per_room_occ": occ,
                                    "responders": n,
                                    "exit_combo": "".join(combo),
                                    "exit_combo_id": combo_id,
                                    "makespan_s": makespan,
                                    "makespan_hms": _fmt(makespan),
                                    "room_order": room_order_str,
                                    "responder_orders": responder_orders,
                                    "start_positions": start_positions,
                                }
                            layout_key = (layout_code, occ)
                            cur_best_layout = best_by_layout_occ.get(layout_key)
                            if cur_best_layout is None or makespan < cur_best_layout["makespan_s"]:
                                best_by_layout_occ[layout_key] = {
                                    "makespan_s": makespan,
                                    "makespan_hms": _fmt(makespan),
                                    "floors": floors,
                                    "responders": n,
                                    "exit_combo": "".join(combo),
                                    "room_order": room_order_str,
                                    "responder_orders": responder_orders,
                                    "start_positions": start_positions,
                                }
                            f.flush()
                            try:
                                os.fsync(f.fileno())
                            except OSError:
                                pass

    print(f"Saved CSV: {args.out}")

    # Write select_result CSV: best initial-responder assignment per (layout,floors,responders)
    if best_by_combo:
        os.makedirs(os.path.dirname(select_out) or ".", exist_ok=True)
        with open(select_out, "w", newline="", encoding="utf-8") as sf:
            sw = csv.writer(sf)
            sw.writerow([
                "layout",
                "floors",
                "per_room_occ",
                "responders",
                "exit_combo",
                "makespan_hms",
                "room_clear_order",
                "exit_combo_id",
                "makespan_s",
                "responder_orders",
                "start_positions",
            ])
            for info in best_by_combo.values():
                sw.writerow([
                    info.get("layout", ""),
                    info.get("floors", 0),
                    info.get("per_room_occ", 0),
                    info.get("responders", 0),
                    info.get("exit_combo", ""),
                    info.get("makespan_hms", ""),
                    info.get("room_order", ""),
                    info.get("exit_combo_id", ""),
                    info.get("makespan_s", 0.0),
                    info.get("responder_orders", ""),
                    info.get("start_positions", ""),
                ])
        print(f"Saved select CSV: {select_out}")

    # Write summary CSV: one row per (layout, per_room_occ) with best over floors/responders/combos
    if best_by_layout_occ:
        # stable sorting: layout by preferred order (BASELINE, T, L) if present, then others; occ asc
        layout_order = {name: i for i, name in enumerate(["BASELINE", "T", "L"]) }
        def _layout_rank(name: str) -> int:
            return layout_order.get(name, 100 + hash(name) % 1000)

        items = sorted(best_by_layout_occ.items(), key=lambda kv: (_layout_rank(kv[0][0]), kv[0][1]))
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "w", newline="", encoding="utf-8") as sf:
            sw = csv.writer(sf)
            sw.writerow([
                "layout",
                "per_room_occ",
                "best_makespan_s",
                "best_makespan_hms",
                "best_floors",
                "best_responders",
                "best_exit_combo",
                "best_room_clear_order",
                "best_responder_orders",
                "best_start_positions",
            ])
            for (layout_code, occ), info in items:
                sw.writerow([
                    layout_code,
                    occ,
                    info["makespan_s"],
                    info["makespan_hms"],
                    info["floors"],
                    info["responders"],
                    info["exit_combo"],
                    info.get("room_order", ""),
                    info.get("responder_orders", ""),
                    info.get("start_positions", ""),
                ])
        print(f"Saved summary CSV: {args.summary_out}")


if __name__ == "__main__":
    main()
