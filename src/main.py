"""Unified entrypoint: ILP sweep (default) + legacy sim mode.

Default: runs ILP-based planner on layout JSON (layout/baseline.json) and
produces console output + PNGs + GIF.

Set RUN_MODE=legacy to run the previous step-based simulation.
Optionally set LAYOUT_FILE to point to another JSON layout.
"""

import os
import sys
import json


def run_ilp():
    # Ensure relative imports
    sys.path.append(os.path.dirname(__file__))

    from ilp_sweep import solve_building_sweep
    from viz import plot_paths, plot_gantt, animate_gif

    # Load layout JSON (default baseline)
    def load_from_layout_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            J = json.load(f)
        J_layout = J.get("layout") if isinstance(J.get("layout"), dict) else J
        # Grid schema
        if "doors" in J_layout:
            xs = list(J_layout.get("doors", {}).get("xs", []))
            topZ = float(J_layout.get("doors", {}).get("topZ"))
            bottomZ = float(J_layout.get("doors", {}).get("bottomZ"))
            room_ids = []
            coords = {}
            for i, x in enumerate(xs):
                rid = f"R{i}"
                room_ids.append(rid)
                coords[rid] = (float(x), float(topZ))
            n_top = len(xs)
            for i, x in enumerate(xs):
                rid = f"R{n_top + i}"
                room_ids.append(rid)
                coords[rid] = (float(x), float(bottomZ))
            frame = J_layout.get("frame", {})
            x1 = float(frame.get("x1"))
            x2 = float(frame.get("x2"))
            cor = J_layout.get("corridor", {})
            cz = float(cor.get("z"))
            ch = float(cor.get("h"))
            midZ = cz + (ch - 1.0) / 2.0
            start_node = {"F1": "E_L", "F2": "E_R"}
            coords["E_L"] = (x1, midZ)
            coords["E_R"] = (x2, midZ)
            per_room = int(J_layout.get("occupants", {}).get("per_room", 0))
            occupants = {rid: per_room for rid in room_ids}
            return room_ids, ["F1", "F2"], start_node, coords, occupants
        # Simple schema
        rooms_raw = list(J_layout.get("rooms", []))
        if rooms_raw:
            room_ids = [r.get("id") for r in rooms_raw]
            coords = {r.get("id"): (float(r.get("entry", r.get("pos", 0.0))), 0.0) for r in rooms_raw}
            responders_raw = J.get("responders", [])
            if responders_raw:
                responders = [r.get("id") for r in responders_raw]
                start_node = {rid: f"E_{rid}" for rid in responders}
                for r in responders_raw:
                    coords[start_node[r.get("id")]] = (float(r.get("pos", 0.0)), 0.0)
            else:
                responders = ["F1", "F2"]
                start_node = {"F1": "E_F1", "F2": "E_F2"}
                xs = [coords[r][0] for r in room_ids]
                coords[start_node["F1"]] = (min(xs) - 5.0, 0.0)
                coords[start_node["F2"]] = (max(xs) + 5.0, 0.0)
            occupants = {r.get("id"): len(r.get("occupants", []) or []) for r in rooms_raw}
            return room_ids, responders, start_node, coords, occupants
        raise SystemExit("Unsupported layout JSON schema")

    scenario_file = os.environ.get(
        "LAYOUT_FILE",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", "baseline.json"),
    )

    if not os.path.exists(scenario_file):
        raise SystemExit(f"Layout file not found: {scenario_file}")

    rooms, responders, start_node, coords, occupants = load_from_layout_json(scenario_file)
    print(f"[layout] Loaded layout from {scenario_file} with {len(rooms)} rooms.")

    # Solve with empirical low-visibility guide + redundant sweep by default
    result = solve_building_sweep(
        rooms=rooms,
        responders=responders,
        start_node=start_node,
        coords=coords,
        occupants=occupants,
        base_check_time=5.0,
        time_per_occupant=3.0,
        walk_speed=1.0,
        verbose=False,
        empirical_mode="guide_low",
        redundancy_mode="per_responder_all_rooms",
        occupant_speed_high=0.5,
        occupant_speed_low=0.3,
        responder_speed_search=0.5,
        responder_speed_carry=0.25,
        carry_capacity=3,
        comm_success=0.85,
    )

    # Print summary similar to ilp_sweep __main__
    from datetime import timedelta

    def format_time(seconds: float) -> str:
        if seconds is None:
            return "N/A"
        td = timedelta(seconds=round(seconds))
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print("=== ILP Result ===")
    print("Status:", result["status"])
    print("Optimal total time (s):", result["T_total"])
    print("Optimal total time (HH:MM:SS):", format_time(result["T_total"]))

    for k, info in result["responders"].items():
        print(f"\nResponder {k}:")
        print("  Time (s):", info["T"])
        print("  Time (HH:MM:SS):", format_time(info["T"]))
        print("  Assigned rooms:", info["assigned_rooms"])
        print("  Path:")
        for edge in info["path"]:
            print("    ", edge[0], "->", edge[1])

    print("\n=== Global summary ===")
    print("Total sweep time (makespan):", format_time(result["T_total"]))
    for k, info in result["responders"].items():
        edges = info["path"]
        if not edges:
            continue
        seq = [edges[0][0]]
        for (_, dst) in edges:
            seq.append(dst)
        print(f"{k} order: " + " -> ".join(seq))

    print("\nGlobal first-clear order (by time):")
    first = result.get("first_clear", {})
    order = first.get("order", [])
    by_room = first.get("by_room", {})
    for r in order:
        rec = by_room.get(r, {})
        t = rec.get("time")
        by = rec.get("by")
        print(f"  {r}: {format_time(t)} by {by}")

    # Visuals
    print("\nGenerating figures under ./out ...")
    plot_paths(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/topo_paths.png")
    plot_gantt(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/gantt.png")
    animate_gif(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/anim.gif", fps=24, duration_sec=12)
    print("Saved: out/topo_paths.png, out/gantt.png, out/anim.gif")

    return result


def run_legacy():
    # Keep the previous step-based loop available if needed
    from configs import load_config
    from utils import init_environment, update_hazard_field, all_cleared, log_status
    from planner import select_next_room, select_next_exit, move, check_room

    cfg = load_config()
    env = init_environment(cfg)
    responders, layout, occupants, hazard = env["responders"], env["layout"], env["occupants"], env["hazard"]

    t = 0
    logs = []

    while not all_cleared(occupants):
        for responder in responders:
            next_room = select_next_room(responder, layout)
            move(responder, next_room, layout)
            status = check_room(responder, next_room, t, hazard)
            log_status(logs, responder, next_room, status, t)
        hazard = update_hazard_field(hazard, dt=cfg["dt"])
        t += cfg["dt"]

    for responder in responders:
        exit_node = select_next_exit(responder, layout)
        move(responder, exit_node, layout)

    T_total = max(r["t_finish"] for r in responders)
    print(f"Total sweep time: {T_total:.2f}")
    return T_total, logs


if __name__ == "__main__":
    mode = os.environ.get("RUN_MODE", "ilp").lower()
    if mode == "legacy":
        run_legacy()
    else:
        run_ilp()
