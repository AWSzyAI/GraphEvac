# 任务分配、路径优化、多agent策略
# src/planner.py
from utils import (
    detect_occupants, trigger_rescue, update_confidence,
    find_safe_path, follow_path, all_occupants_evacuated
)

def check_room(Rj, room, t, hazard):
    """Top-level room check procedure"""
    # --- Step 1: Doorway Observation ---
    local = doorway_observation(Rj, room, hazard)
    if local == "occupied":
        return "occupied"
    if local == "unobserved":
        return "pending"

    # --- Step 2: Attempt Entry ---
    local = attempt_entry(Rj, room, hazard)
    if local == "deferred":
        return "deferred"
    if local == "occupied":
        return "occupied"

    # --- Step 3: Judge Final State ---
    if all_occupants_evacuated(room):
        room["status"] = "cleared"
        room["t_cleared"] = t
        return "cleared"
    return "pending"


def doorway_observation(Rj, room, hazard):
    """Local visual scan at doorway"""
    d_eff = Rj["view"] * pow(2.71828, -Rj["alpha"] * hazard[room["id"]])
    dist = abs(Rj["pos"] - room["pos"])
    if dist > d_eff:
        return "unobserved"

    room["observed"] = True
    if detect_occupants(Rj, room):
        trigger_rescue(Rj, room)
        room["confidence"] = 0
        return "occupied"
    else:
        update_confidence(room, "obs")
        return "observed"


def attempt_entry(Rj, room, hazard):
    """Attempt to enter and inspect"""
    path = find_safe_path(Rj["pos"], room["entry"], hazard)
    if path is None or max(hazard.get(p, 0) for p in path) > Rj["H_limit"]:
        room["blocked"] = True
        return "deferred"

    follow_path(Rj, path)
    if detect_occupants(Rj, room):
        trigger_rescue(Rj, room)
        room["confidence"] = 0
        return "occupied"

    update_confidence(room, "entry")
    return "pending"
