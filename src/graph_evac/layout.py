"""Layout loading helpers."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple


Coord = Tuple[float, float]


@dataclass
class Layout:
    rooms: List[str]
    responders: List[str]
    start_nodes: Dict[str, str]
    coords: Dict[str, Coord]
    occupants: Dict[str, int]


def _coord_from(value) -> Coord:
    if isinstance(value, dict):
        return (float(value.get("x", 0.0)), float(value.get("z", 0.0)))
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (float(value[0]), 0.0)
        return (float(value[0]), float(value[1]))
    return (float(value), 0.0)


def load_layout(path: str | Path) -> Layout:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    layout_data = raw.get("layout") if isinstance(raw, dict) else raw
    if layout_data is None and isinstance(raw, dict):
        layout_data = raw
    if not isinstance(layout_data, dict):
        raise ValueError("Layout JSON must contain a 'layout' object or be the layout itself")

    if "doors" in layout_data:
        xs = list(layout_data.get("doors", {}).get("xs", []))
        top_z = float(layout_data.get("doors", {}).get("topZ", 0.0))
        bottom_z = float(layout_data.get("doors", {}).get("bottomZ", 0.0))
        room_ids: List[str] = []
        coords: Dict[str, Coord] = {}
        for idx, x in enumerate(xs):
            rid = f"R{idx}"
            room_ids.append(rid)
            coords[rid] = (float(x), top_z)
        for idx, x in enumerate(xs):
            rid = f"R{len(xs) + idx}"
            room_ids.append(rid)
            coords[rid] = (float(x), bottom_z)
        frame = layout_data.get("frame", {})
        cor = layout_data.get("corridor", {})
        start_nodes = {"F1": "E_L", "F2": "E_R"}
        coords["E_L"] = (float(frame.get("x1", 0.0)), float(cor.get("z", 0.0)))
        coords["E_R"] = (float(frame.get("x2", 0.0)), float(cor.get("z", 0.0)))
        per_room = int(layout_data.get("occupants", {}).get("per_room", 0))
        occupants = {rid: per_room for rid in room_ids}
        responders = ["F1", "F2"]
        return Layout(room_ids, responders, start_nodes, coords, occupants)

    rooms_raw = list(layout_data.get("rooms", []))
    room_ids = [room.get("id") for room in rooms_raw]
    coords = {}
    for room in rooms_raw:
        rid = room.get("id")
        coords[rid] = _coord_from(room.get("entry", room.get("pos", 0.0)))

    responders_raw = raw.get("responders", []) if isinstance(raw, dict) else []
    if responders_raw:
        responders = [resp.get("id") for resp in responders_raw]
        start_nodes = {resp.get("id"): f"E_{resp.get('id')}" for resp in responders_raw}
        for resp in responders_raw:
            coords[start_nodes[resp.get("id")]] = _coord_from(resp.get("pos", 0.0))
    else:
        responders = ["F1", "F2"]
        xs = [coords[rid][0] for rid in room_ids]
        start_nodes = {"F1": "E_F1", "F2": "E_F2"}
        coords[start_nodes["F1"]] = (min(xs) - 5.0, 0.0)
        coords[start_nodes["F2"]] = (max(xs) + 5.0, 0.0)

    occupants: Dict[str, int] = {}
    for room in rooms_raw:
        rid = room.get("id")
        occ_val = room.get("occupants", [])
        if isinstance(occ_val, list):
            occupants[rid] = len(occ_val)
        else:
            try:
                occupants[rid] = int(occ_val)
            except Exception:
                occupants[rid] = 0

    return Layout(room_ids, responders, start_nodes, coords, occupants)


def expand_floors(layout: Layout, floors: int, spacing: float = 60.0) -> tuple[Layout, List[str]]:
    if floors <= 1:
        exits = sorted(set(layout.start_nodes.values()))
        return layout, exits

    new_coords = {k: v for k, v in layout.coords.items() if k not in layout.rooms}
    new_rooms: List[str] = []
    new_occ: Dict[str, int] = {}

    for floor in range(floors):
        z_offset = floor * spacing
        for rid in layout.rooms:
            coord = layout.coords.get(rid, (0.0, 0.0))
            new_id = f"{rid}_F{floor + 1}"
            new_rooms.append(new_id)
            new_coords[new_id] = (float(coord[0]), float(coord[1]) + z_offset)
            new_occ[new_id] = int(layout.occupants.get(rid, 0))

    exit_variants: List[str] = []
    for exit_name in sorted(set(layout.start_nodes.values())):
        coord = layout.coords.get(exit_name, (0.0, 0.0))
        for floor in range(floors):
            eid = f"{exit_name}_F{floor + 1}"
            new_coords[eid] = (float(coord[0]), float(coord[1]) + floor * spacing)
            exit_variants.append(eid)

    start_nodes = {rid: f"{exit_name}_F1" for rid, exit_name in layout.start_nodes.items()}
    expanded = Layout(new_rooms, layout.responders, start_nodes, new_coords, new_occ)
    return expanded, exit_variants
