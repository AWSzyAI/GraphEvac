# 任务分配、路径优化、多agent策略
# src/planner.py
from logging_utils import get_logger
from typing import Optional
from utils import (
    detect_occupants,
    trigger_rescue,
    update_confidence,
    find_safe_path,
    follow_path,
    all_occupants_evacuated,
)

def check_room(Rj, room, t, hazard):
    """Top-level room check procedure"""
    logger = get_logger()
    # --- Step 1: Doorway Observation ---
    local = doorway_observation(Rj, room, hazard)
    logger.debug("[check_room] %s @doorway %s -> %s", Rj.get("id"), room.get("id"), local)
    if local == "occupied":
        return "occupied"
    if local == "unobserved":
        return "pending"

    # --- Step 2: Attempt Entry ---
    local = attempt_entry(Rj, room, hazard)
    logger.debug("[check_room] %s @entry %s -> %s", Rj.get("id"), room.get("id"), local)
    if local == "deferred":
        return "deferred"
    if local == "occupied":
        return "occupied"

    # --- Step 3: Judge Final State ---
    if all_occupants_evacuated(room):
        room["status"] = "cleared"
        room["t_cleared"] = t
        logger.info("[check_room] %s CLEARED by %s at t=%s", room.get("id"), Rj.get("id"), t)
        return "cleared"
    return "pending"


def doorway_observation(Rj, room, hazard):
    """Local visual scan at doorway"""
    logger = get_logger()
    d_eff = Rj["view"] * pow(2.71828, -Rj["alpha"] * hazard[room["id"]])
    dist = abs(Rj["pos"] - room["pos"])
    if dist > d_eff:
        logger.debug("[doorway] %s too far from %s (dist=%.2f > d_eff=%.2f)", Rj.get("id"), room.get("id"), dist, d_eff)
        return "unobserved"

    room["observed"] = True
    if detect_occupants(Rj, room):
        trigger_rescue(Rj, room)
        room["confidence"] = 0
        logger.info("[doorway] %s detects occupants in %s -> RESCUE", Rj.get("id"), room.get("id"))
        return "occupied"
    else:
        update_confidence(room, "obs")
        logger.debug("[doorway] %s observes %s -> no occupant, confidence=%.2f", Rj.get("id"), room.get("id"), room.get("confidence", 0))
        return "observed"


def attempt_entry(Rj, room, hazard):
    """Attempt to enter and inspect"""
    logger = get_logger()
    path = find_safe_path(Rj["pos"], room["entry"], hazard)
    # Use room-level hazard thresholding (path is a stub of numeric positions)
    room_h = hazard.get(room["id"], 0)
    if path is None or room_h > Rj.get("H_limit", 1.0):
        room["blocked"] = True
        logger.warning("[entry] %s cannot enter %s (hazard=%.2f > H_limit=%.2f) -> deferred", Rj.get("id"), room.get("id"), room_h, Rj.get("H_limit", 1.0))
        return "deferred"

    follow_path(Rj, path)
    if detect_occupants(Rj, room):
        trigger_rescue(Rj, room)
        room["confidence"] = 0
        logger.info("[entry] %s finds occupants in %s -> RESCUE", Rj.get("id"), room.get("id"))
        return "occupied"

    update_confidence(room, "entry")
    logger.debug("[entry] %s scanned %s -> pending, confidence=%.2f", Rj.get("id"), room.get("id"), room.get("confidence", 0))
    return "pending"


# -------------------------------
# Legacy planner helpers (selection/move)
# -------------------------------

def _room_is_cleared(room: dict) -> bool:
    return room.get("status") == "cleared"


def _room_entry_pos(room: dict) -> float:
    return float(room.get("entry", room.get("pos", 0.0)))


def _distance_to_room(Rj: dict, room: dict) -> float:
    pos = float(Rj.get("pos", 0.0))
    return abs(_room_entry_pos(room) - pos)


def ensure_clear_queue(Rj: dict, layout: dict) -> None:
    if "clear_queue" in Rj:
        return
    rooms = list(layout.get("rooms", []))
    rooms_sorted = sorted(rooms, key=lambda r: _distance_to_room(Rj, r))
    Rj["clear_queue"] = rooms_sorted


def select_next_room(Rj: dict, layout: dict) -> Optional[dict]:
    """Return the next room this responder has yet to clear."""
    ensure_clear_queue(Rj, layout)
    queue = Rj.get("clear_queue", [])
    if not queue:
        return None
    return queue[0]


def select_next_exit(Rj: dict, layout: dict) -> dict:
    """Choose the nearest synthetic exit beyond room endpoints."""
    rooms = layout.get("rooms", [])
    if not rooms:
        return {"id": f"E_{Rj.get('id','X')}", "entry": Rj.get("pos", 0.0)}
    xs = [float(r.get("entry", r.get("pos", 0.0))) for r in rooms]
    left = min(xs) - 5.0
    right = max(xs) + 5.0
    pos = float(Rj.get("pos", 0.0))
    goal = left if abs(pos - left) <= abs(pos - right) else right
    return {"id": f"E_{Rj.get('id','X')}", "entry": goal}


def move(Rj, target, layout, hazard=None):
    """Move responder to target entry using simple path stub."""
    goal = float(target.get("entry", target.get("pos", 0.0)))
    start = float(Rj.get("pos", 0.0))
    path = find_safe_path(start, goal, hazard or {})
    follow_path(Rj, path)
