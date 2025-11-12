# src/utils.py
import random
from logging_utils import get_logger

def init_environment(cfg):
    layout = cfg["layout"]
    responders = cfg["responders"]
    occupants = cfg["occupants"]
    hazard = cfg["hazard"]
    logger = get_logger()
    logger.info("[legacy:init] rooms=%d responders=%d occupants=%d dt=%s", len(layout.get("rooms", [])), len(responders), len(occupants), cfg.get("dt"))
    return {"layout": layout, "responders": responders, "occupants": occupants, "hazard": hazard}


def detect_occupants(Rj, room):
    """Simple probabilistic detection"""
    occs = [o for o in room["occupants"] if o["status"] == "inside"]
    visible = [o for o in occs if random.random() < Rj["detection_prob"]]
    return len(visible) > 0


def trigger_rescue(Rj, room):
    """Handle occupant rescue (evacuate a single occupant per trip)."""
    for o in room["occupants"]:
        if o["status"] == "inside":
            o["status"] = "evacuated"
            return o


def update_confidence(room, mode="obs"):
    """Incremental information completeness"""
    if "confidence" not in room:
        room["confidence"] = 0
    delta = {"obs": 0.2, "entry": 0.5, "sensor": 0.1}.get(mode, 0.1)
    room["confidence"] = min(1.0, room["confidence"] + delta)


def find_safe_path(start, goal, hazard):
    """Placeholder pathfinding (mocked as always available)"""
    return [start, goal]


def follow_path(Rj, path):
    """Move responder along the path"""
    logger = get_logger()
    logger.debug("[move] %s: %s -> %s (steps=%d)", Rj.get("id"), Rj.get("pos"), path[-1], len(path))
    Rj["pos"] = path[-1]
    Rj["t_finish"] = Rj.get("t_finish", 0) + len(path)


def update_hazard_field(hazard, dt=1.0):
    """Simplified hazard diffusion model"""
    logger = get_logger()
    for r in hazard:
        hazard[r] = min(1.0, hazard[r] + 0.01 * dt)
    logger.debug("[hazard] updated with dt=%.2f", dt)
    return hazard


def all_occupants_evacuated(room):
    return all(o["status"] in ("evacuated", "dead") for o in room["occupants"])


def all_cleared(occupants):
    return all(o["status"] in ("evacuated", "dead") for o in occupants)


def log_status(logs, responder, room, status, t):
    logs.append({
        "time": t,
        "responder": responder["id"],
        "room": room["id"],
        "status": status
    })
