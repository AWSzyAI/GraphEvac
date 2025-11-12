"""Unified entrypoint: ILP sweep (default) + legacy sim mode.

Default: runs ILP-based planner on layout JSON (layout/baseline.json) and
produces console output + PNGs + GIF.

Set RUN_MODE=legacy to run the previous step-based simulation.
Optionally set LAYOUT_FILE to point to another JSON layout.
"""

import json
import os
import sys
from pathlib import Path

from logging_utils import add_file_handler, setup_logging


def run_ilp():
    logger = setup_logging(debug=os.environ.get("RUN_DEBUG", "0") == "1")
    # Ensure relative imports
    sys.path.append(os.path.dirname(__file__))

    from greedy_sweep import solve_building_sweep
    from viz import animate_gif, plot_gantt, plot_paths

    def load_from_layout_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            J = json.load(f)
        J_layout = J.get("layout") if isinstance(J.get("layout"), dict) else J
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

        rooms_raw = list(J_layout.get("rooms", []))
        if rooms_raw:
            def _coord_from(v):
                if isinstance(v, dict):
                    return (float(v.get("x", 0.0)), float(v.get("z", 0.0)))
                return (float(v), 0.0)

            room_ids = [r.get("id") for r in rooms_raw]
            coords = {}
            for r in rooms_raw:
                rid = r.get("id")
                entry = r.get("entry", r.get("pos", 0.0))
                coords[rid] = _coord_from(entry)
            responders_raw = J.get("responders", [])
            if responders_raw:
                responders = [r.get("id") for r in responders_raw]
                start_node = {rid: f"E_{rid}" for rid in responders}
                for r in responders_raw:
                    coords[start_node[r.get("id")]] = _coord_from(r.get("pos", 0.0))
            else:
                responders = ["F1", "F2"]
                start_node = {"F1": "E_F1", "F2": "E_F2"}
                xs = [coords[r][0] for r in room_ids]
                coords[start_node["F1"]] = (min(xs) - 5.0, 0.0)
                coords[start_node["F2"]] = (max(xs) + 5.0, 0.0)
            occupants = {}
            for r in rooms_raw:
                rid = r.get("id")
                occ_val = r.get("occupants", [])
                if isinstance(occ_val, list):
                    occupants[rid] = len(occ_val)
                else:
                    try:
                        occupants[rid] = int(occ_val)
                    except Exception:
                        occupants[rid] = 0
            return room_ids, responders, start_node, coords, occupants

        raise SystemExit("Unsupported layout JSON schema")

    scenario_file = os.environ.get(
        "LAYOUT_FILE",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", "baseline.json"),
    )

    try:
        from configs import SIM_CONFIG as _SIM
    except Exception:
        _SIM = {}

    if not os.path.exists(scenario_file):
        raise SystemExit(f"Layout file not found: {scenario_file}")

    rooms, responders, start_node, coords, occupants = load_from_layout_json(scenario_file)

    floors_cfg = int(os.environ.get("FLOORS", _SIM.get("floors", 1)))
    floor_spacing = float(_SIM.get("floor_spacing", 60.0))
    rooms, coords, occupants, start_node, exit_nodes = _expand_floors(
        rooms=rooms,
        coords=coords,
        occupants=occupants,
        start_node=start_node,
        floors=floors_cfg,
        spacing=floor_spacing,
    )

    try:
        with open(scenario_file, "r", encoding="utf-8") as _f:
            _Jroot = json.load(_f)
        _Jsim = _Jroot.get("sim", {}) if isinstance(_Jroot, dict) else {}
        if isinstance(_Jsim, dict) and _Jsim:
            _SIM = {**_SIM, **_Jsim}
    except Exception:
        _Jsim = {}

    eff_cfg = {
        "layout_file": scenario_file,
        "redundancy_mode": _SIM.get("redundancy_mode", "per_responder_all_rooms"),
        "empirical_mode": _SIM.get("empirical_mode", "guide_low"),
        "base_check_time": _SIM.get("base_check_time", 5.0),
        "time_per_occupant": _SIM.get("time_per_occupant", 3.0),
        "walk_speed": _SIM.get("walk_speed", 1.0),
        "occupant_speed_high": _SIM.get("occupant_speed_high", 0.5),
        "occupant_speed_low": _SIM.get("occupant_speed_low", 0.3),
        "responder_speed_search": _SIM.get("responder_speed_search", 0.5),
        "responder_speed_carry": _SIM.get("responder_speed_carry", 0.25),
        "carry_capacity": _SIM.get("carry_capacity", 3),
        "comm_success": _SIM.get("comm_success", 0.85),
        "egress_distance_factor": _SIM.get("egress_distance_factor", 1.0),
        "responder_view": _SIM.get("responder_view", 15.0),
        "responder_alpha": _SIM.get("responder_alpha", 0.2),
        "responder_H_limit": _SIM.get("responder_H_limit", 0.7),
        "responder_detection_prob": _SIM.get("responder_detection_prob", 1.0),
        "responders": list(responders),
        "start_node": dict(start_node),
        "room_count": len(rooms),
        "total_occupants": sum(int(occupants.get(r, 0)) for r in rooms),
        "run_mode": "ilp",
        "floors": floors_cfg,
        "floor_spacing": floor_spacing,
        "exits": exit_nodes,
    }

    def _sanitize_segment(value: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)

    layout_label = _sanitize_segment(Path(scenario_file).stem)
    resp_label = _sanitize_segment("-".join(responders))
    redundancy_label = _sanitize_segment(eff_cfg["redundancy_mode"])
    empirical_label = _sanitize_segment(str(eff_cfg["empirical_mode"] or "default"))
    floors_label = f"floors_{eff_cfg.get('floors', 1)}"
    occ_label = f"occ_{eff_cfg.get('total_occupants', 0)}"
    tag = f"layout_{layout_label}_{floors_label}_resp_{resp_label}_{occ_label}_red_{redundancy_label}_emp_{empirical_label}"
    output_root = Path(os.environ.get("OUTPUT_ROOT", "out"))
    output_dir = output_root / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "run.log")
    logger.info("[layout] Loaded %d rooms from %s", len(rooms), scenario_file)
    logger.info("[output] Saving visualization artifacts under %s", output_dir)
    logger.info("[params] %s", eff_cfg)
    logger.info("[config] Effective simulation configuration:")
    config_order = [
        "base_check_time",
        "carry_capacity",
        "comm_success",
        "egress_distance_factor",
        "empirical_mode",
        "layout_file",
        "occupant_speed_high",
        "occupant_speed_low",
        "redundancy_mode",
        "responder_speed_carry",
        "responder_speed_search",
        "responders",
        "room_count",
        "run_mode",
        "start_node",
        "time_per_occupant",
        "total_occupants",
        "walk_speed",
        "responder_view",
        "responder_alpha",
        "responder_H_limit",
        "responder_detection_prob",
        "floors",
        "floor_spacing",
    ]
    for key in config_order:
        if key in eff_cfg:
            logger.info("  - %s: %s", key, eff_cfg[key])

    result = solve_building_sweep(
        rooms=rooms,
        responders=responders,
        start_node=start_node,
        coords=coords,
        occupants=occupants,
        base_check_time=eff_cfg["base_check_time"],
        time_per_occupant=eff_cfg["time_per_occupant"],
        walk_speed=eff_cfg["walk_speed"],
        verbose=False,
        empirical_mode=eff_cfg["empirical_mode"],
        redundancy_mode=eff_cfg["redundancy_mode"],
        occupant_speed_high=eff_cfg["occupant_speed_high"],
        occupant_speed_low=eff_cfg["occupant_speed_low"],
        responder_speed_search=eff_cfg["responder_speed_search"],
        responder_speed_carry=eff_cfg["responder_speed_carry"],
        carry_capacity=eff_cfg["carry_capacity"],
        comm_success=eff_cfg["comm_success"],
        egress_distance_factor=eff_cfg["egress_distance_factor"],
    )

    # 将可用出口传给可视化层
    try:
        result.setdefault("meta", {})["exits"] = eff_cfg.get("exits", list(start_node.values()))
    except Exception:
        pass

    sim_layout, sim_responders, sim_hazard = _build_sim_environment(
        rooms=rooms,
        coords=coords,
        start_node=start_node,
        occupants=occupants,
        sim_cfg=eff_cfg,
    )
    _simulate_rescue_plan(
        logger,
        sim_layout,
        sim_responders,
        sim_hazard,
        result,
        coords,
        start_node,
        eff_cfg,
    )

    from datetime import timedelta

    def format_time(seconds: float) -> str:
        if seconds is None:
            return "N/A"
        td = timedelta(seconds=round(seconds))
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    logger.info("=== ILP Result ===")
    logger.info("Status: %s", result["status"])
    logger.info("Optimal total time (s): %s", result["T_total"])
    logger.info("Optimal total time (HH:MM:SS): %s", format_time(result["T_total"]))

    for k, info in result["responders"].items():
        logger.info("")
        logger.info("Responder %s:", k)
        logger.info("  Time (s): %s", info["T"])
        logger.info("  Time (HH:MM:SS): %s", format_time(info["T"]))
        logger.info("  Assigned rooms: %s", info["assigned_rooms"])
        logger.info("  Path:")
        for edge in info["path"]:
            logger.info("    %s -> %s", edge[0], edge[1])

    logger.info("")
    logger.info("=== Global summary ===")
    logger.info("Total sweep time (makespan): %s", format_time(result["T_total"]))
    for k, info in result["responders"].items():
        edges = info.get("path", [])
        if not edges:
            continue
        seq = [edges[0][0]]
        for (_, dst) in edges:
            seq.append(dst)
        logger.info("%s order: %s", k, " -> ".join(seq))

    first = result.get("first_clear", {})
    order = first.get("order", [])
    by_room = first.get("by_room", {})
    logger.info("")
    logger.info("Global first-clear order (by time):")
    for r in order:
        rec = by_room.get(r, {})
        t = rec.get("time")
        by = rec.get("by")
        logger.info("  %s: %s by %s", r, format_time(t), by)
    if order:
        logger.info("Full first-clear order: %s", " -> ".join(order))

    topo_path = output_dir / "topo_paths.png"
    gantt_path = output_dir / "gantt.png"
    gif_path = output_dir / "anim.gif"
    logger.info("")
    logger.info("Generating figures under %s ...", output_dir)
    plot_paths(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath=str(topo_path))
    plot_gantt(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath=str(gantt_path))
    animate_gif(
        coords=coords,
        rooms=rooms,
        start_node=start_node,
        result=result,
        savepath=str(gif_path),
        fps=24,
        duration_sec=12,
    )
    logger.info("Saved: %s, %s, %s", topo_path, gantt_path, gif_path)

    return result


def _build_sim_environment(rooms, coords, start_node, occupants, sim_cfg):
    layout_rooms = []
    hazard = {}
    for rid in rooms:
        coord = coords.get(rid, 0.0)
        x = coord[0] if isinstance(coord, (tuple, list)) else float(coord)
        occ_count = int(occupants.get(rid, 0))
        layout_rooms.append(
            {
                "id": rid,
                "pos": x,
                "entry": x,
                "occupants": [{"id": f"{rid}_o{i+1}", "status": "inside"} for i in range(occ_count)],
            }
        )
        hazard[rid] = 0.1
    layout = {"rooms": layout_rooms}
    responders_env = []
    for rid, node in start_node.items():
        coord = coords.get(node, coords.get(rid, 0.0))
        x = coord[0] if isinstance(coord, (tuple, list)) else float(coord)
        responders_env.append(
            {
                "id": rid,
                "pos": x,
                "view": sim_cfg.get("responder_view", 15.0),
                "alpha": sim_cfg.get("responder_alpha", 0.2),
                "H_limit": sim_cfg.get("responder_H_limit", 0.7),
                "detection_prob": sim_cfg.get("responder_detection_prob", 1.0),
            }
        )
    return layout, responders_env, hazard


def _expand_floors(rooms, coords, occupants, start_node, floors, spacing=60.0):
    if floors <= 1:
        exits = sorted(set(start_node.values()))
        return rooms, coords, occupants, start_node, exits

    base_rooms = list(rooms)
    new_rooms = []
    new_coords = {k: v for k, v in coords.items() if k not in base_rooms}
    new_occ = {}

    for floor in range(floors):
        z_offset = floor * spacing
        for rid in base_rooms:
            coord = coords.get(rid, (0.0, 0.0))
            x = coord[0] if isinstance(coord, (tuple, list)) else float(coord)
            z = coord[1] if isinstance(coord, (tuple, list)) and len(coord) > 1 else 0.0
            new_id = f"{rid}_F{floor+1}"
            new_rooms.append(new_id)
            new_coords[new_id] = (float(x), float(z) + z_offset)
            new_occ[new_id] = int(occupants.get(rid, 0))

    unique_exits = sorted(set(start_node.values()))
    exit_variants = []
    new_start = {}
    for exit_name in unique_exits:
        coord = coords.get(exit_name, (0.0, 0.0))
        x = coord[0] if isinstance(coord, (tuple, list)) else float(coord)
        z = coord[1] if isinstance(coord, (tuple, list)) and len(coord) > 1 else 0.0
        for floor in range(floors):
            eid = f"{exit_name}_F{floor+1}"
            new_coords[eid] = (float(x), float(z) + floor * spacing)
            exit_variants.append(eid)
    for rid, exit_name in start_node.items():
        new_start[rid] = f"{exit_name}_F1"

    return new_rooms, new_coords, new_occ, new_start, exit_variants


def _simulate_rescue_plan(logger, layout, responders, hazard, result, coords, start_node, sim_cfg):
    from planner import check_room
    from utils import log_status, update_hazard_field
    import heapq

    room_map = {room["id"]: room for room in layout["rooms"]}
    sweep_time = result.get("sweep_time", {})
    timeline_map = {responder["id"]: [] for responder in responders}
    search_speed = sim_cfg.get("responder_speed_search", 0.5)
    carry_speed = sim_cfg.get("responder_speed_carry", 0.25)

    def _fmt_time(seconds: float) -> str:
        from datetime import timedelta
        td = timedelta(seconds=round(seconds))
        h, rem = divmod(td.seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _node_coord(node: str):
        coord = coords.get(node)
        if coord is None:
            room = room_map.get(node)
            if room is not None:
                base = room.get("entry", room.get("pos", 0.0))
                return (float(base), 0.0)
            return (0.0, 0.0)
        if isinstance(coord, (tuple, list)):
            return (float(coord[0]), float(coord[1]))
        return (float(coord), 0.0)

    def _node_distance(a: str, b: str) -> float:
        p0 = _node_coord(a)
        p1 = _node_coord(b)
        dx = p0[0] - p1[0]
        dy = p0[1] - p1[1]
        return (dx * dx + dy * dy) ** 0.5

    def _record_segment(rid, typ, node_from, node_to, t0, t1):
        timeline_map[rid].append(
            {
                "type": typ,
                "from": node_from,
                "to": node_to,
                "p0": _node_coord(node_from),
                "p1": _node_coord(node_to),
                "t0": t0,
                "t1": t1,
            }
        )

    def _floor_tag(node: str) -> int:
        if "_F" in node:
            try:
                return int(node.rsplit("_F", 1)[1])
            except Exception:
                return 1
        return 1

    def _is_exit(node: str) -> bool:
        return node in exit_candidates or node.startswith("E_")

    def _exits_on_floor(floor: int):
        return [e for e in exit_candidates if _floor_tag(e) == floor]

    # build plans (list of nodes) for each responder
    plans = {}
    for responder in result.get("responders", {}):
        edges = result["responders"][responder].get("path", [])
        seq = [start_node.get(responder, responder)]
        for (_, v) in edges:
            seq.append(v)
        plans[responder] = seq

    # shared room occupant pool
    occ_pool = {room["id"]: len(room.get("occupants", [])) for room in layout["rooms"]}

    exit_candidates = list(sim_cfg.get("exits", list(start_node.values())))

    def _nearest_exit(node: str) -> str:
        if not exit_candidates:
            return node
        best = exit_candidates[0]
        best_dist = _node_distance(node, best)
        for cand in exit_candidates[1:]:
            d = _node_distance(node, cand)
            if d < best_dist:
                best = cand
                best_dist = d
        return best

    # priority queue for concurrent simulation
    heap = []
    states = {}
    for responder in responders:
        rid = responder["id"]
        states[rid] = {
            "idx": 0,
            "time": 0.0,
            "current": start_node.get(rid, rid),
            "plan": plans.get(rid, [start_node.get(rid, rid)]),
            "home": start_node.get(rid, rid),
        }
        heapq.heappush(heap, (0.0, rid))

    def _need_floor_transition(a: str, b: str) -> bool:
        def _floor_tag(node: str):
            if "_F" in node:
                try:
                    return int(node.rsplit("_F", 1)[1])
                except Exception:
                    return 1
            return 1
        return _floor_tag(a) != _floor_tag(b)

    while heap:
        t_cur, rid = heapq.heappop(heap)
        state = states[rid]
        plan = state["plan"]
        idx = state["idx"]
        if idx >= len(plan) - 1:
            # ensure responder returns home if not already there
            target = _nearest_exit(state["current"])
            if state["current"] != target:
                dist = _node_distance(state["current"], target)
                dt = dist / max(search_speed, 1e-6)
                _record_segment(rid, "walk", state["current"], target, t_cur, t_cur + dt)
                state["current"] = target
                state["time"] = t_cur + dt
                logger.info("[log %s] %s returned to exit %s", _fmt_time(state["time"]), rid, target)
            continue

        nxt = plan[idx + 1]
        move_to = nxt
        # map dummy G nodes to actual exits
        if nxt.startswith("G_"):
            move_to = _nearest_exit(state["current"])

        if _need_floor_transition(state["current"], move_to) and not _is_exit(state["current"]) and not _is_exit(move_to):
            cf = _floor_tag(state["current"])
            tf = _floor_tag(move_to)
            # current -> nearest exit on current floor
            ex_cands = _exits_on_floor(cf) or exit_candidates
            ex_c = min(ex_cands, key=lambda e: _node_distance(state["current"], e))
            d1 = _node_distance(state["current"], ex_c)
            dt1 = d1 / max(search_speed, 1e-6)
            _record_segment(rid, "walk", state["current"], ex_c, t_cur, t_cur + dt1)
            t_cur += dt1
            # stairs: exit_cf -> exit_tf
            ex_t_cands = _exits_on_floor(tf) or exit_candidates
            ex_t = min(ex_t_cands, key=lambda e: _node_distance(ex_c, e))
            d2 = _node_distance(ex_c, ex_t)
            dt2 = d2 / max(search_speed, 1e-6)
            _record_segment(rid, "stairs", ex_c, ex_t, t_cur, t_cur + dt2)
            t_cur += dt2
            # exit_tf -> target room
            d3 = _node_distance(ex_t, move_to)
            dt3 = d3 / max(search_speed, 1e-6)
            _record_segment(rid, "walk", ex_t, move_to, t_cur, t_cur + dt3)
            t_cur += dt3
            state["current"] = move_to
        else:
            dist = _node_distance(state["current"], move_to)
            dt_walk = dist / max(search_speed, 1e-6)
            typ = "stairs" if _need_floor_transition(state["current"], move_to) else "walk"
            _record_segment(rid, typ, state["current"], move_to, t_cur, t_cur + dt_walk)
            state["current"] = move_to
            t_cur += dt_walk
        state["time"] = t_cur
        state["idx"] += 1

        if move_to in occ_pool:
            room_id = move_to
            occ = occ_pool[room_id]
            while occ > 0:
                exit_target = _nearest_exit(room_id)
                logger.info("[log %s] %s rescues occupant in %s -> %s", _fmt_time(t_cur), rid, room_id, exit_target)
                dist_exit = _node_distance(room_id, exit_target)
                dt_out = dist_exit / max(carry_speed, 1e-6)
                _record_segment(rid, "escort", room_id, exit_target, t_cur, t_cur + dt_out)
                t_cur += dt_out
                occ -= 1
                occ_pool[room_id] = occ
                logger.info("[log %s] %s delivered occupant to %s", _fmt_time(t_cur), rid, exit_target)
                # return to room if occupants remain
                if occ > 0:
                    dt_back = dist_exit / max(search_speed, 1e-6)
                    typ_back = "stairs" if _need_floor_transition(exit_target, room_id) else "return"
                    _record_segment(rid, typ_back, exit_target, room_id, t_cur, t_cur + dt_back)
                    t_cur += dt_back
            # final verification dwell
            dwell = float(sweep_time.get(room_id, sim_cfg.get("base_check_time", 5.0)))
            if dwell > 0:
                _record_segment(rid, "clear", room_id, room_id, t_cur, t_cur + dwell)
                t_cur += dwell
            logger.info("[log %s] %s cleared %s", _fmt_time(t_cur), rid, room_id)
            state["current"] = room_id
            state["time"] = t_cur

        heapq.heappush(heap, (state["time"], rid))
        state["time"] = t_cur

        hazard = update_hazard_field(hazard, dt=1.0)

    for rid in timeline_map:
        result["responders"].setdefault(rid, {})["timeline"] = timeline_map[rid]
    return result


def run_legacy():
    logger = setup_logging(debug=os.environ.get("RUN_DEBUG", "0") == "1")
    from configs import load_config
    from planner import (
        check_room,
        ensure_clear_queue,
        move,
        select_next_exit,
        select_next_room,
    )
    from utils import init_environment, log_status, update_hazard_field

    cfg = load_config()
    env = init_environment(cfg)
    responders, layout, occupants, hazard = env["responders"], env["layout"], env["occupants"], env["hazard"]

    for responder in responders:
        ensure_clear_queue(responder, layout)

    def _has_pending_rooms() -> bool:
        return any(responder.get("clear_queue") for responder in responders)

    t = 0
    logs = []
    logger.info("[legacy] Starting legacy loop with %d responders and %d occupants", len(responders), len(occupants))
    while _has_pending_rooms():
        for responder in responders:
            queue = responder.get("clear_queue", [])
            if not queue:
                continue
            next_room = select_next_room(responder, layout)
            if next_room is None:
                continue
            queue.pop(0)
            logger.debug("t=%s | %s -> next room %s", t, responder.get("id"), next_room.get("id"))
            move(responder, next_room, layout)
            status = check_room(responder, next_room, t, hazard)
            log_status(logs, responder, next_room, status, t)
            if status != "cleared":
                queue.append(next_room)
                logger.debug("t=%s | re-queue %s for %s (status=%s)", t, next_room.get("id"), responder.get("id"), status)
            logger.debug("t=%s | %s checked %s => %s", t, responder.get("id"), next_room.get("id"), status)
            if status == "occupied":
                exit_node = select_next_exit(responder, layout)
                logger.info(
                    "[rescue] %s escorting occupants from %s to %s",
                    responder.get("id"),
                    next_room.get("id"),
                    exit_node.get("id"),
                )
                move(responder, exit_node, layout)
                logger.info(
                    "[rescue] %s delivered occupants to %s and returns to queue",
                    responder.get("id"),
                    exit_node.get("id"),
                )
        hazard = update_hazard_field(hazard, dt=cfg["dt"])
        t += cfg["dt"]

    for responder in responders:
        exit_node = select_next_exit(responder, layout)
        move(responder, exit_node, layout)

    T_total = max(r.get("t_finish", 0) for r in responders)
    logger.info("[legacy] Total sweep time: %.2f", T_total)
    return T_total, logs


if __name__ == "__main__":
    mode = os.environ.get("RUN_MODE", "ilp").lower()
    if mode == "legacy":
        run_legacy()
    else:
        run_ilp()
