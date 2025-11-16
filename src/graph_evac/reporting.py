"""Post-run reporting helpers for GraphEvac."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import Config
from .problem import EvacuationProblem


CSV_FIELDS = [
    "timestamp",
    "layout",
    "floors",
    "responders_number",
    "responders_init_position",
    "order",
    "time",
]


def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _walk_speed(config: Config) -> float:
    return max(float(getattr(config, "walk_speed", 1.0) or 0.0), 1e-6)


def _extract_visit_order(path: Iterable[Tuple[str, str]], rooms: Iterable[str]) -> List[str]:
    room_set = set(rooms)
    order: List[str] = []
    for (_, target) in path or []:
        if target in room_set:
            order.append(target)
    return order


def _convert_timelines(problem: EvacuationProblem, timeline: Dict[str, List[dict]] | None) -> Dict[str, List[dict]]:
    if not timeline:
        return {}
    converted: Dict[str, List[dict]] = {}
    for rid, entries in timeline.items():
        seq: List[dict] = []
        for entry in entries:
            typ = entry.get("type")
            if typ == "walk":
                start = entry.get("from")
                end = entry.get("to")
                if not start or not end:
                    continue
                seq.append(
                    {
                        "type": "walk",
                        "from": start,
                        "to": end,
                        "t0": float(entry.get("t0", 0.0)),
                        "t1": float(entry.get("t1", entry.get("t0", 0.0))),
                        "p0": problem.node_coord(start),
                        "p1": problem.node_coord(end),
                    }
                )
            elif typ == "inspect":
                room = entry.get("room")
                if not room:
                    continue
                seq.append(
                    {
                        "type": "clear",
                        "from": room,
                        "to": room,
                        "t0": float(entry.get("t0", 0.0)),
                        "t1": float(entry.get("t1", entry.get("t0", 0.0))),
                        "p0": problem.node_coord(room),
                        "p1": problem.node_coord(room),
                    }
                )
        converted[rid] = seq
    return converted


def _resolve_makespan(plan_data: dict, timeline: Dict[str, List[dict]] | None) -> float:
    if timeline:
        max_t = 0.0
        for entries in timeline.values():
            for entry in entries:
                try:
                    max_t = max(max_t, float(entry.get("t1", 0.0)))
                except (TypeError, ValueError):
                    continue
        if max_t > 0.0:
            return max_t
    try:
        return float(plan_data.get("makespan", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_first_clear(
    problem: EvacuationProblem,
    plan_data: dict,
    timeline: Dict[str, List[dict]] | None,
    config: Config,
) -> Tuple[Dict[str, dict], List[str]]:
    first: Dict[str, dict] = {}
    if timeline:
        for rid, entries in timeline.items():
            for entry in entries:
                if entry.get("type") != "inspect":
                    continue
                room = entry.get("room")
                if not room:
                    continue
                t_clear = float(entry.get("t1", entry.get("t0", 0.0)))
                rec = first.get(room)
                if rec is None or t_clear < rec.get("time", float("inf")):
                    first[room] = {"time": t_clear, "by": rid}
    if not first:
        walk_speed = _walk_speed(config)
        for rid, info in plan_data.get("responders", {}).items():
            t = 0.0
            start_node = problem.responders[rid].start_node if rid in problem.responders else None
            current = start_node
            for (u, v) in info.get("path", []):
                origin = u or current
                dest = v or origin
                t += problem.distance(origin, dest) / walk_speed
                if dest in problem.rooms:
                    t += problem.room_service_time(dest)
                    rec = first.get(dest)
                    if rec is None or t < rec.get("time", float("inf")):
                        first[dest] = {"time": t, "by": rid}
                current = dest
    order = sorted(first, key=lambda rid: first[rid]["time"])
    return first, order


def summarize_run(
    problem: EvacuationProblem,
    plan_data: dict,
    timeline: Dict[str, List[dict]] | None,
    config: Config,
) -> Dict[str, object]:
    if not plan_data:
        return {}
    room_ids = list(problem.room_ids)
    responder_orders = {
        rid: _extract_visit_order(info.get("path", []), room_ids)
        for rid, info in plan_data.get("responders", {}).items()
    }
    first_clear, room_order = _build_first_clear(problem, plan_data, timeline, config)
    return {
        "first_clear": first_clear,
        "room_order": room_order,
        "responder_orders": responder_orders,
        "makespan": _resolve_makespan(plan_data, timeline),
        "viz_timelines": _convert_timelines(problem, timeline),
    }


def build_viz_payload(
    problem: EvacuationProblem,
    plan_data: dict,
    timeline: Dict[str, List[dict]] | None,
    config: Config,
) -> dict:
    summary = summarize_run(problem, plan_data, timeline, config)
    if not summary:
        return {}
    responders_payload: Dict[str, dict] = {}
    for rid, info in plan_data.get("responders", {}).items():
        payload = {
            "path": info.get("path", []),
            "rooms": summary["responder_orders"].get(rid, []),
            "time": info.get("time"),
        }
        timeline_seq = summary["viz_timelines"].get(rid)
        if timeline_seq:
            payload["timeline"] = timeline_seq
        responders_payload[rid] = payload
    sweep_time = {room_id: problem.room_service_time(room_id) for room_id in problem.room_ids}
    return {
        "responders": responders_payload,
        "first_clear": {"by_room": summary["first_clear"], "order": summary["room_order"]},
        "sweep_time": sweep_time,
        "meta": {
            "walk_speed_used": config.walk_speed,
            "redundancy_mode": config.redundancy_mode,
            "exits": list(problem.exits),
            "layout_file": config.layout_file,
            "floors": config.floors,
        },
        "T_total": summary["makespan"],
    }


def summarize_run(
    problem: EvacuationProblem,
    plan_data: dict,
    timeline: Dict[str, List[dict]] | None,
    config: Config,
) -> Dict[str, object]:
    if not plan_data:
        return {}
    room_ids = list(problem.room_ids)
    responder_orders = {
        rid: _extract_visit_order(info.get("path", []), room_ids)
        for rid, info in plan_data.get("responders", {}).items()
    }
    first_clear, room_order = _build_first_clear(problem, plan_data, timeline, config)
    return {
        "first_clear": first_clear,
        "room_order": room_order,
        "responder_orders": responder_orders,
        "makespan": _resolve_makespan(plan_data, timeline),
        "viz_timelines": _convert_timelines(problem, timeline),
    }


def build_result_record(
    problem: EvacuationProblem,
    plan_data: dict,
    timeline: Dict[str, List[dict]] | None,
    config: Config,
    output_dir: Path,
) -> Dict[str, object]:
    summary = summarize_run(problem, plan_data, timeline, config)
    if not summary:
        return {}
    config_dict = asdict(config)
    start_positions = [f"{rid}:{resp.start_node}" for rid, resp in problem.responders.items()]
    responders_number = len(problem.responders)
    room_clear_order = "->".join(summary["room_order"]) if summary["room_order"] else "N/A"
    makespan = float(summary.get("makespan", 0.0))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "layout": Path(config.layout_file).stem,
        "floors": config.floors,
        "responders_number": responders_number,
        "responders_init_position": ";".join(start_positions),
        "order": room_clear_order,
        "time": seconds_to_hms(makespan),
    }
    return record


def append_result_record(record: Dict[str, object], csv_path: Path) -> None:
    if not record:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_header(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})


def _ensure_header(csv_path: Path) -> None:
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
        return
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = reader.fieldnames
        if existing == CSV_FIELDS:
            return
        rows = list(reader)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
