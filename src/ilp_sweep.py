#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ILP building sweep demo
- Responder = F (消防员)
- Occupants = 被救的人
- sweep_time 由每个 room 的 occupant 数量决定

依赖:
    pip install pulp
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ILP building sweep demo (baseline layout version)
"""

import pulp as pl
import json
from datetime import timedelta



def build_travel_time(coords, walk_speed=1.0):
    """
    根据坐标生成任意两点之间的行走时间。

    coords 的 value 可以是:
      - 标量 (1D)，例如 x
      - 长度为 2 的 tuple/list (2D)，例如 (x, z)

    参数
    ----
    coords : dict[str, float | (float,float)]
        节点 -> 坐标
    walk_speed : float
        行走速度 (m/s)

    返回
    ----
    travel_time : dict[(str, str), float]
        (i, j) -> 从 i 到 j 的时间
    """
    travel_time = {}
    for i, vi in coords.items():
        for j, vj in coords.items():
            if i == j:
                continue

            # 1D / 2D 自适应
            if isinstance(vi, (tuple, list)):
                dx = vi[0] - vj[0]
                dz = vi[1] - vj[1]
                distance = (dx ** 2 + dz ** 2) ** 0.5
            else:
                distance = abs(vi - vj)

            travel_time[(i, j)] = distance / walk_speed
    return travel_time


def _euclid(a, b):
    if isinstance(a, (tuple, list)):
        dx = a[0] - b[0]
        dz = a[1] - b[1]
        return (dx * dx + dz * dz) ** 0.5
    return abs(a - b)


def compute_room_sweep_time(
    rooms,
    coords,
    exit_nodes,
    occupants,
    base_check_time=5.0,
    empirical_mode=None,
    # Occupant speeds (m/s)
    occupant_speed_high=0.5,
    occupant_speed_low=0.3,
    # Responder speeds (m/s)
    responder_speed_search=0.5,
    responder_speed_carry=0.25,
    # Carry/comm behavior
    carry_capacity=3,
    comm_success=0.85,
):
    """
    Compute per-room sweep time using empirical assumptions.

    empirical_mode options:
      - None: use only base_check_time (no per-occupant overhead)
      - 'guide_high': occupant guided at high visibility
      - 'guide_low' : occupant guided at low visibility
      - 'carry'     : occupants carried/dragged in groups

    Notes
    -----
    - Distances use straight-line metric over provided coords.
    - For guiding, effective pair speed is limited by both parties:
        v = min(occupant_speed, responder_speed_search)
      Per-occupant overhead ~= distance_to_nearest_exit / v.
    - For carrying, we approximate grouped shuttles back-and-forth:
        groups = ceil(N / carry_capacity)
        time_carry = groups * (2 * distance_to_nearest_exit / responder_speed_carry)
      This is added once per room (not per occupant).
    - Communication success affects time multiplicatively: divide by p.
    """
    # Precompute shortest distance from each room to any exit
    exits = list(exit_nodes)
    dist_to_exit = {}
    for r in rooms:
        dmin = min(_euclid(coords[r], coords[e]) for e in exits)
        dist_to_exit[r] = dmin

    import math

    sweep_time = {}
    for r in rooms:
        N = int(occupants.get(r, 0))
        t_room = base_check_time

        if empirical_mode is None or N == 0:
            sweep_time[r] = t_room
            continue

        d = dist_to_exit[r]

        if empirical_mode == "guide_high":
            v_occ = min(occupant_speed_high, responder_speed_search)
            per_occ = d / max(v_occ, 1e-6)
            t_room += (per_occ * N) / max(comm_success, 1e-6)
        elif empirical_mode == "guide_low":
            v_occ = min(occupant_speed_low, responder_speed_search)
            per_occ = d / max(v_occ, 1e-6)
            t_room += (per_occ * N) / max(comm_success, 1e-6)
        elif empirical_mode == "carry":
            groups = math.ceil(N / max(int(carry_capacity), 1))
            shuttle = (2.0 * d) / max(responder_speed_carry, 1e-6)
            t_room += (groups * shuttle) / max(comm_success, 1e-6)
        else:
            # Fallback to no empirical overhead
            pass

        sweep_time[r] = t_room

    return sweep_time


def solve_building_sweep(
    rooms,
    responders,
    start_node,
    coords,
    occupants,
    base_check_time=5.0,
    time_per_occupant=3.0,
    walk_speed=1.0,
    *,
    empirical_mode=None,
    occupant_speed_high=0.5,
    occupant_speed_low=0.3,
    responder_speed_search=0.5,
    responder_speed_carry=0.25,
    carry_capacity=3,
    comm_success=0.85,
    redundancy_mode="assignment",
    solver=None,
    verbose=False,
):
    """
    求解最优 sweep order 和总时间的 ILP 模型。

    参数
    ----
    rooms : list[str]
        房间列表，例如 ["R0", "R1", ...]
    responders : list[str]
        Responder 列表，例如 ["F1", "F2"]
    start_node : dict[str, str]
        responder -> 起点节点名，例如 {"F1": "E_L", "F2": "E_R"}
    coords : dict[str, float|(float,float)]
        所有节点坐标（可 1D 或 2D）
    occupants : dict[str, int]
        room -> occupant 数量
    base_check_time : float
        每个房间的固定检查时间 (秒)
    time_per_occupant : float
        每个 occupant 额外花费时间 (秒/人)
    walk_speed : float
        行走速度 (m/s)
    solver : pulp solver or None
        不传则用默认 CBC
    verbose : bool
        是否打印求解器 log

    返回
    ----
    result : dict
        {
          "status": 求解状态,
          "T_total": 最优总时间,
          "responders": {
              "F1": {
                  "T": 时间,
                  "assigned_rooms": [...],
                  "path": [(node_i, node_j), ...]
              },
              ...
          }
        }
    """
    # If empirical responder speeds are set, default walk_speed to search speed
    if empirical_mode is not None and walk_speed == 1.0:
        walk_speed = responder_speed_search

    exit_nodes = set(start_node.values())

    # 由 occupant 数量得到每个房间的 sweep_time
    # Use empirical per-room times if requested; otherwise simple linear model
    if empirical_mode is None:
        sweep_time = {r: base_check_time + time_per_occupant * occupants.get(r, 0) for r in rooms}
    else:
        sweep_time = compute_room_sweep_time(
            rooms=rooms,
            coords=coords,
            exit_nodes=exit_nodes,
            occupants=occupants,
            base_check_time=base_check_time,
            empirical_mode=empirical_mode,
            occupant_speed_high=occupant_speed_high,
            occupant_speed_low=occupant_speed_low,
            responder_speed_search=responder_speed_search,
            responder_speed_carry=responder_speed_carry,
            carry_capacity=carry_capacity,
            comm_success=comm_success,
        )

    # ---------- 建立模型 ----------
    prob = pl.LpProblem("Building_Sweep_ILP", pl.LpMinimize)

    n = len(rooms)

    if redundancy_mode == "per_responder_all_rooms":
        # 构建每个 responder 的节点：Start_k, Rooms, End_k(dummy)
        # travel_time_k 包含 s_k->room, room->room, room->g_k
        def min_return_time(r_name):
            dmin = min(_euclid(coords[r_name], coords[e]) for e in exit_nodes)
            return dmin / walk_speed

        y_keys = []
        travel_time_k = {}
        nodes_k = {}
        for k in responders:
            s = start_node[k]
            g = f"G_{k}"
            nodes_k[k] = ([s] + rooms + [g])
            # s_k -> rooms
            for j in rooms:
                y_keys.append((k, s, j))
                travel_time_k[(k, s, j)] = _euclid(coords[s], coords[j]) / walk_speed
            # rooms -> rooms
            for i in rooms:
                for j in rooms:
                    if i == j:
                        continue
                    y_keys.append((k, i, j))
                    travel_time_k[(k, i, j)] = _euclid(coords[i], coords[j]) / walk_speed
            # rooms -> g_k (nearest exit)
            for i in rooms:
                y_keys.append((k, i, g))
                travel_time_k[(k, i, g)] = min_return_time(i)

        # 决策变量
        y = pl.LpVariable.dicts("arc", y_keys, lowBound=0, upBound=1, cat=pl.LpBinary)
        u = pl.LpVariable.dicts(
            "order",
            [(k, r) for k in responders for r in rooms],
            lowBound=1,
            upBound=n,
            cat=pl.LpContinuous,
        )
        T = pl.LpVariable.dicts("T", responders, lowBound=0, cat=pl.LpContinuous)
        T_total = pl.LpVariable("T_total", lowBound=0, cat=pl.LpContinuous)

        # 目标
        prob += T_total, "Minimize_total_completion_time"

        # 约束：起点一条出边；终点一条入边；每间房间入=1、出=1
        for k in responders:
            s = start_node[k]
            g = f"G_{k}"
            # Start out = 1
            prob += (
                pl.lpSum(y[(k, s, j)] for j in rooms) == 1,
                f"start_from_{k}",
            )
            # End in = 1
            prob += (
                pl.lpSum(y[(k, i, g)] for i in rooms) == 1,
                f"end_into_{k}",
            )
            # Rooms in/out = 1
            for r in rooms:
                prob += (
                    pl.lpSum([y[(k, i, r)] for i in rooms if i != r] + [y[(k, s, r)]]) == 1,
                    f"room_in_{k}_{r}",
                )
                prob += (
                    pl.lpSum([y[(k, r, j)] for j in rooms if j != r] + [y[(k, r, g)]]) == 1,
                    f"room_out_{k}_{r}",
                )

            # MTZ subtour elimination over rooms only
            M = n
            for i in rooms:
                for j in rooms:
                    if i == j:
                        continue
                    prob += (
                        u[(k, i)] - u[(k, j)] + M * y[(k, i, j)] <= M - 1,
                        f"mtz_{k}_{i}_{j}",
                    )

        # 每个 responder 的总时间：行走 + 每房间清扫时间（每人都要清一遍）
        for k in responders:
            walk_term = pl.lpSum(travel_time_k[(k, i, j)] * y[(k, i, j)] for (kk, i, j) in y_keys if kk == k)
            sweep_term = pl.lpSum(sweep_time[r] for r in rooms)
            prob += (T[k] >= walk_term + sweep_term, f"time_def_{k}")

        # Makespan
        for k in responders:
            prob += (T_total >= T[k], f"Ttotal_ge_T_{k}")

    else:
        # 原始：分配房间给各 responder，每个房间被清一次
        nodes_k = {k: [start_node[k]] + rooms for k in responders}
        travel_time = build_travel_time(coords, walk_speed=walk_speed)

        # 决策变量
        x = pl.LpVariable.dicts(
            "assign",
            [(r, k) for r in rooms for k in responders],
            lowBound=0,
            upBound=1,
            cat=pl.LpBinary,
        )
        y = pl.LpVariable.dicts(
            "arc",
            [
                (k, i, j)
                for k in responders
                for i in nodes_k[k]
                for j in nodes_k[k]
                if i != j
            ],
            lowBound=0,
            upBound=1,
            cat=pl.LpBinary,
        )
        u = pl.LpVariable.dicts(
            "order",
            [(k, r) for k in responders for r in rooms],
            lowBound=0,
            upBound=n,
            cat=pl.LpContinuous,
        )
        T = pl.LpVariable.dicts("T", responders, lowBound=0, cat=pl.LpContinuous)
        T_total = pl.LpVariable("T_total", lowBound=0, cat=pl.LpContinuous)

        # 目标
        prob += T_total, "Minimize_total_completion_time"

        # 约束
        for r in rooms:
            prob += (pl.lpSum(x[(r, k)] for k in responders) == 1, f"one_responder_per_room_{r}")

        for k in responders:
            s_node = start_node[k]
            prob += (
                pl.lpSum(y[(k, s_node, j)] for j in nodes_k[k] if j != s_node) == 1,
                f"start_from_exit_{k}",
            )

        for r in rooms:
            prob += (
                pl.lpSum(y[(k, i, r)] for k in responders for i in nodes_k[k] if i != r) == 1,
                f"one_incoming_{r}",
            )

        for k in responders:
            for r in rooms:
                prob += (
                    pl.lpSum(y[(k, r, j)] for j in nodes_k[k] if j != r) == x[(r, k)],
                    f"outdeg_matches_assign_{r}_{k}",
                )
                prob += (
                    pl.lpSum(y[(k, i, r)] for i in nodes_k[k] if i != r) <= x[(r, k)],
                    f"incoming_le_assign_{r}_{k}",
                )

        M = n
        for k in responders:
            for r in rooms:
                prob += (u[(k, r)] >= x[(r, k)], f"order_lb_{k}_{r}")
                prob += (u[(k, r)] <= n * x[(r, k)], f"order_ub_{k}_{r}")
            for i in rooms:
                for j in rooms:
                    if i == j:
                        continue
                    prob += (u[(k, i)] - u[(k, j)] + M * y[(k, i, j)] <= M - 1, f"mtz_{k}_{i}_{j}")

        for k in responders:
            walk_term = pl.lpSum(
                travel_time[(i, j)] * y[(k, i, j)]
                for i in nodes_k[k]
                for j in nodes_k[k]
                if i != j
            )
            sweep_term = pl.lpSum(sweep_time[r] * x[(r, k)] for r in rooms)
            prob += (T[k] >= walk_term + sweep_term, f"time_def_{k}")

        for k in responders:
            prob += (T_total >= T[k], f"Ttotal_ge_T_{k}")

    # ---------- 求解 ----------
    if solver is None:
        solver = pl.PULP_CBC_CMD(msg=verbose)

    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    result = {
        "status": status,
        "T_total": pl.value(T_total),
        "responders": {},
        "meta": {
            "redundancy_mode": redundancy_mode,
            "empirical_mode": empirical_mode,
            "walk_speed_used": walk_speed,
        },
        "sweep_time": sweep_time,
    }

    # ---------- 解析结果 ----------
    # 收集每个房间的首次 clear 时间（跨 responder 取最早）
    room_first_clear_time = {r: None for r in rooms}
    room_first_clear_by = {r: None for r in rooms}

    for k in responders:
        if redundancy_mode == "per_responder_all_rooms":
            assigned_rooms = list(rooms)
            s = start_node[k]
            g = f"G_{k}"
            # 从起点顺着 y 找路径，直到到达 g
            path = []
            current = s
            visited = {current}
            # 构造候选节点集合（起点+房间+终点）
            node_candidates = [s] + rooms + [g]
            while True:
                next_nodes = [
                    j
                    for j in node_candidates
                    if j != current and (k, current, j) in y and pl.value(y[(k, current, j)]) > 0.5
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                path.append((current, nxt))
                if nxt == g:
                    break
                if nxt in visited:
                    break
                visited.add(nxt)
                current = nxt
        else:
            assigned_rooms = [r for r in rooms if pl.value(x[(r, k)]) > 0.5]
            # 从起点开始顺着 y 变量把路径走一遍（得到 sweep order）
            path = []
            current = start_node[k]
            visited = {current}
            while True:
                next_nodes = [
                    j
                    for j in nodes_k[k]
                    if j != current and pl.value(y[(k, current, j)]) > 0.5
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                path.append((current, nxt))
                if nxt in visited:
                    break
                visited.add(nxt)
                current = nxt

        # 基于路径累积行走+清扫时间，得到此 responder 的房间 clear 时间线
        t = 0.0
        clear_times_k = {}
        for (u_node, v_node) in path:
            # 行走时间
            if redundancy_mode == "per_responder_all_rooms":
                dt_walk = 0.0
                if (k, u_node, v_node) in locals().get('travel_time_k', {}):
                    dt_walk = travel_time_k[(k, u_node, v_node)]
                else:
                    # 退化计算（不应发生）：
                    dt_walk = _euclid(coords[u_node], coords[v_node]) / max(walk_speed, 1e-6)
            else:
                dt_walk = build_travel_time(coords, walk_speed=walk_speed).get((u_node, v_node), 0.0)
            t += dt_walk

            # 进入房间后执行清扫
            if v_node in rooms:
                dt_sweep = sweep_time[v_node]
                t_clear = t + dt_sweep
                clear_times_k[v_node] = t_clear
                # 更新全局“首次 clear”
                if room_first_clear_time[v_node] is None or t_clear < room_first_clear_time[v_node]:
                    room_first_clear_time[v_node] = t_clear
                    room_first_clear_by[v_node] = k
                # 停留清扫
                t += dt_sweep

        result["responders"][k] = {
            "T": pl.value(T[k]),
            "assigned_rooms": assigned_rooms,
            "path": path,
            "room_clear_times": clear_times_k,
        }

    # 生成全局首次 clear 顺序
    first_clear_items = [
        (r, room_first_clear_time[r], room_first_clear_by[r])
        for r in rooms
        if room_first_clear_time[r] is not None
    ]
    first_clear_items.sort(key=lambda x: x[1])
    result["first_clear"] = {
        "by_room": {r: {"time": t, "by": k} for (r, t, k) in first_clear_items},
        "order": [r for (r, _, _) in first_clear_items],
    }

    return result






if __name__ == "__main__":
    # Ensure local src directory is on path (for venv runs)
    import os, sys
    sys.path.append(os.path.dirname(__file__))

    # 优先：从 layout JSON 读取（默认 layout/baseline.json，可用 LAYOUT_FILE 环境覆盖）
    def load_from_layout_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            J = json.load(f)
        # Accept two schemas:
        # A) Grid baseline: doors/topZ/bottomZ/frame/corridor/occupants.per_room
        # B) Simple linear: layout.rooms[{id,pos,entry,occupants}], responders[{id,pos,...}]

        # If wrapped under 'layout', unwrap
        J_layout = J.get("layout") if isinstance(J.get("layout"), dict) else J

        if "doors" in J_layout:
            xs = list(J_layout.get("doors", {}).get("xs", []))
            topZ = float(J_layout.get("doors", {}).get("topZ"))
            bottomZ = float(J_layout.get("doors", {}).get("bottomZ"))
            room_ids = []
            room_coords = {}
            for i, x in enumerate(xs):
                rid = f"R{i}"
                room_ids.append(rid)
                room_coords[rid] = (float(x), float(topZ))
            n_top = len(xs)
            for i, x in enumerate(xs):
                rid = f"R{n_top + i}"
                room_ids.append(rid)
                room_coords[rid] = (float(x), float(bottomZ))
            frame = J_layout.get("frame", {})
            x1 = float(frame.get("x1"))
            x2 = float(frame.get("x2"))
            cor = J_layout.get("corridor", {})
            cz = float(cor.get("z"))
            ch = float(cor.get("h"))
            midZ = cz + (ch - 1.0) / 2.0
            start_node_map = {"F1": "E_L", "F2": "E_R"}
            room_coords["E_L"] = (x1, midZ)
            room_coords["E_R"] = (x2, midZ)
            per_room = int(J_layout.get("occupants", {}).get("per_room", 0))
            occ = {rid: per_room for rid in room_ids}
            responders = ["F1", "F2"]
            return room_ids, responders, start_node_map, room_coords, occ

        # Simple linear schema
        rooms_raw = list(J_layout.get("rooms", []))
        if rooms_raw:
            room_ids = [r.get("id") for r in rooms_raw]
            coords = {}
            for r in rooms_raw:
                rid = r.get("id")
                x = r.get("entry", r.get("pos", 0.0))
                coords[rid] = (float(x), 0.0)
            # Responders
            responders_raw = J.get("responders", [])
            if responders_raw:
                responders = [r.get("id") for r in responders_raw]
                start_node_map = {rid: f"E_{rid}" for rid in responders}
                for r in responders_raw:
                    coords[start_node_map[r.get("id")]] = (float(r.get("pos", 0.0)), 0.0)
            else:
                responders = ["F1", "F2"]
                start_node_map = {"F1": "E_F1", "F2": "E_F2"}
                # place at extremes
                xs = [coords[r][0] for r in room_ids]
                coords[start_node_map["F1"]] = (min(xs) - 5.0, 0.0)
                coords[start_node_map["F2"]] = (max(xs) + 5.0, 0.0)
            # Occupants count
            occ = {}
            for r in rooms_raw:
                rid = r.get("id")
                occ_list = r.get("occupants", [])
                occ[rid] = len(occ_list) if isinstance(occ_list, list) else int(occ_list or 0)
            return room_ids, responders, start_node_map, coords, occ

        raise ValueError("Unsupported layout JSON schema")

    scenario_file = os.environ.get("LAYOUT_FILE", os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", "baseline.json"))
    rooms = responders = start_node = coords = occupants = None
    extra = {}
    if os.path.exists(scenario_file):
        try:
            rooms, responders, start_node, coords, occupants = load_from_layout_json(scenario_file)
            print(f"[layout] Loaded layout from {scenario_file} with {len(rooms)} rooms.")
        except Exception as e:
            print("[layout] Failed to load JSON:", e)

    # 次之：从 SIM_CONFIG 读取
    if rooms is None:
        try:
            from configs import SIM_CONFIG  # preferred
            rooms = SIM_CONFIG["rooms"]
            responders = SIM_CONFIG["responders"]
            start_node = SIM_CONFIG["start_node"]
            coords = SIM_CONFIG["coords"]
            occupants = SIM_CONFIG["occupants"]
            extra = SIM_CONFIG
        except Exception:
            # 兼容旧版 configs：提供 load_config() 并自动转换
            try:
                import configs as _cfg
                data = _cfg.load_config()
                # rooms/coords from layout (1D -> embed on z=0)
                rooms = [r["id"] for r in data.get("layout", {}).get("rooms", [])]
                coords = {r["id"]: (float(r.get("pos", 0.0)), 0.0) for r in data.get("layout", {}).get("rooms", [])}
                # responders and synthetic exits at responder pos
                responders = [r["id"] for r in data.get("responders", [])]
                start_node = {r["id"]: f"E_{r['id']}" for r in data.get("responders", [])}
                for r in data.get("responders", []):
                    coords[start_node[r["id"]]] = (float(r.get("pos", 0.0)), 0.0)
                # occupants per room
                occupants = {r["id"]: len(r.get("occupants", [])) for r in data.get("layout", {}).get("rooms", [])}
                extra = {}
                print("[configs] Using legacy load_config() adapter for ilp_sweep.")
            except Exception:
                raise SystemExit("Missing layout file and configs.SIM_CONFIG or compatible load_config().")

    # 解释：
    # - 若 empirical_mode 为 None，则使用线性时间模型: base + N * time_per_occupant
    # - 若启用经验模式，则自动按出口距离和速度估算每个房间的清扫时间

    result = solve_building_sweep(
        rooms=rooms,
        responders=responders,
        start_node=start_node,
        coords=coords,
        occupants=occupants,
        base_check_time=extra.get("base_check_time", 5.0),
        time_per_occupant=extra.get("time_per_occupant", 3.0),
        walk_speed=extra.get("walk_speed", 1.0),
        verbose=False,
        empirical_mode=extra.get("empirical_mode", "guide_low"),
        redundancy_mode=extra.get("redundancy_mode", "per_responder_all_rooms"),
        occupant_speed_high=extra.get("occupant_speed_high", 0.5),
        occupant_speed_low=extra.get("occupant_speed_low", 0.3),
        responder_speed_search=extra.get("responder_speed_search", 0.5),
        responder_speed_carry=extra.get("responder_speed_carry", 0.25),
        carry_capacity=extra.get("carry_capacity", 3),
        comm_success=extra.get("comm_success", 0.85),
    )

    # ---------- 实用小函数：秒 → HH:MM:SS ----------
    def format_time(seconds: float) -> str:
        """把秒转成 HH:MM:SS 格式"""
        if seconds is None:
            return "N/A"
        td = timedelta(seconds=round(seconds))
        # timedelta 默认格式是 "H:MM:SS"
        h, remainder = divmod(td.seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ---------- 打印结果 ----------
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

    # -------- Global summary: 总时间 + 总 order --------
    print("\n=== Global summary ===")
    print("Total sweep time (makespan):", format_time(result["T_total"]))

    # 把每个 F 的 path 转成 node 序列
    for k, info in result["responders"].items():
        edges = info["path"]
        if not edges:
            continue
        seq = [edges[0][0]]  # 起点
        for (_, dst) in edges:
            seq.append(dst)
        print(f"{k} order: " + " -> ".join(seq))

    # 打印全局首次 clear 顺序
    print("\nGlobal first-clear order (by time):")
    first = result.get("first_clear", {})
    order = first.get("order", [])
    by_room = first.get("by_room", {})
    for r in order:
        rec = by_room.get(r, {})
        t = rec.get("time")
        by = rec.get("by")
        print(f"  {r}: {format_time(t)} by {by}")

    # 生成可视化图像
    try:
        from viz import plot_paths, plot_gantt, animate_gif
        print("\nGenerating figures under ./out ...")
        plot_paths(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/topo_paths.png")
        plot_gantt(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/gantt.png")
        animate_gif(coords=coords, rooms=rooms, start_node=start_node, result=result, savepath="out/anim.gif", fps=24, duration_sec=12)
        print("Saved: out/topo_paths.png, out/gantt.png, out/anim.gif")
    except Exception as e:
        print("[viz] Failed to generate figures:", e)
