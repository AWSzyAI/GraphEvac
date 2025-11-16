"""
Simulation configuration for GraphEvac.

All adjustable parameters should live either in this file (configs.py)
or in the JSON layout file (layout/*.json). Geometry is recommended to
live in baseline.json; speeds/modes live here. JSON may optionally
include a "sim" section to override these values per-scenario.
"""

from typing import Dict, Any
import json
import os


# Central simulation settings (used by ILP entry in main.py)
SIM_CONFIG: Dict[str, Any] = {
    # Geometry source
    "layout_file": os.environ.get("LAYOUT_FILE", 
                                  os.path.join(os.path.dirname(os.path.dirname(__file__)), "layout", 
                                               "baseline.json"
                                            #    "layout_L.json"
                                            #    "layout_T.json"
                                               )),

    # Modes
    "redundancy_mode": "per_responder_all_rooms",  # "assignment" | "per_responder_all_rooms"
    "empirical_mode": None,            # None | "guide_high" | "guide_low" | "carry"

    # Timing
    "base_check_time": 30.0,
    "time_per_occupant": 8.0,
    "walk_speed": 0.6,  # if empirical_mode set and ==1.0, search speed will be used

    # Speeds (m/s)
    "occupant_speed_high": 0.5,
    "occupant_speed_low": 0.3,   # 0.2–0.4 typical low-visibility
    "responder_speed_search": 0.6,
    "responder_speed_carry": 0.25,

    # Capacity/comm
    "carry_capacity": 3,
    "comm_success": 0.85,
    # Distance scaling for escort models (1.0 = to exit; <1.0 = to corridor/door)
    "egress_distance_factor": 0.3,
    # Responder sensory defaults (used by legacy sim + ILP replay)
    "responder_view": 15.0,
    "responder_alpha": 0.2,
    "responder_H_limit": 0.7,
    "responder_detection_prob": 1.0,
}


# === Batch sweep defaults (used by `make batch`) ===
# Adjust this block to control the default ranges/layouts/parameters
# that `make batch` uses when you don't override them via env vars.
BATCH_CONFIG: Dict[str, Any] = {
    "floors": "1-18",
    "layouts": "BASELINE,T,L",
    "occ": "5-10",
    "resp": "1-10",
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mk_occ(J_layout: Dict[str, Any]):
    per = int(J_layout.get("occupants", {}).get("per_room", 0))
    return [{"id": f"o{i+1}", "status": "inside"} for i in range(per)]


def load_config() -> Dict[str, Any]:
    """
    Legacy simulator config (RUN_MODE=legacy). Builds a simple 1D layout
    from the same baseline.json so room count stays consistent.
    """
    layout_path = SIM_CONFIG["layout_file"]
    J = _load_json(layout_path)
    J_layout = J.get("layout") if isinstance(J.get("layout"), dict) else J

    rooms = []
    view = SIM_CONFIG.get("responder_view", 15.0)
    alpha = SIM_CONFIG.get("responder_alpha", 0.2)
    H_limit = SIM_CONFIG.get("responder_H_limit", 0.7)
    detection_prob = SIM_CONFIG.get("responder_detection_prob", 1.0)
    if "doors" in J_layout:
        # Grid schema → two rows of rooms, map to 1D by x
        xs = list(J_layout.get("doors", {}).get("xs", []))
        # Top row
        for i, x in enumerate(xs):
            rid = f"R{i}"
            rooms.append({"id": rid, "pos": float(x), "entry": float(x), "occupants": _mk_occ(J_layout)})
        # Bottom row
        n_top = len(xs)
        for i, x in enumerate(xs):
            rid = f"R{n_top + i}"
            rooms.append({"id": rid, "pos": float(x), "entry": float(x), "occupants": _mk_occ(J_layout)})
        frame = J_layout.get("frame", {})
        x1 = float(frame.get("x1")); x2 = float(frame.get("x2"))
        responders = [
            {"id": "A", "pos": x1, "view": view, "alpha": alpha, "H_limit": H_limit, "detection_prob": detection_prob},
            {"id": "B", "pos": x2, "view": view, "alpha": alpha, "H_limit": H_limit, "detection_prob": detection_prob},
        ]
    else:
        # Simple schema (already 1D rooms)
        rooms = list(J_layout.get("rooms", []))
        responders = J.get("responders", [
            {"id": "A", "pos": min([r.get("pos", 0.0) for r in rooms]) - 5.0, "view": view, "alpha": alpha, "H_limit": H_limit, "detection_prob": detection_prob},
            {"id": "B", "pos": max([r.get("pos", 0.0) for r in rooms]) + 5.0, "view": view, "alpha": alpha, "H_limit": H_limit, "detection_prob": detection_prob},
        ])

    layout = {"rooms": rooms}
    occupants = [o for r in rooms for o in r.get("occupants", [])]
    hazard = {r["id"]: 0.1 for r in rooms}
    return {"layout": layout, "responders": responders, "occupants": occupants, "hazard": hazard, "dt": 1.0}
