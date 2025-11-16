"""New CLI entrypoint that wires configuration, planning and simulation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from configs import SIM_CONFIG
from graph_evac import (
    Config,
    EvacuationProblem,
    expand_floors,
    load_layout,
    plan_sweep,
    simulate_sweep,
)
from graph_evac.io_utils import prepare_output_dir, save_plan, save_timeline
from graph_evac.reporting import (
    append_result_record,
    build_result_record,
    build_viz_payload,
    summarize_run,
)
from graph_evac.visualizer import export_visuals
from logging_utils import add_file_handler, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphEvac planner")
    parser.add_argument("--layout", default=Config().layout_file, help="Path to the layout JSON file")
    parser.add_argument(
        "--redundancy-mode",
        default=Config().redundancy_mode,
        choices=["assignment", "per_responder_all_rooms", "double_check"],
        help="Redundancy strategy used by the planner",
    )
    parser.add_argument("--floors", type=int, default=Config().floors, help="Number of floors to expand the layout to")
    parser.add_argument(
        "--floor-spacing",
        type=float,
        default=Config().floor_spacing,
        help="Vertical spacing applied when duplicating floors",
    )
    parser.add_argument("--output-root", default=Config().output_root, help="Directory for generated artifacts")
    parser.add_argument("--no-sim", action="store_true", help="Skip the timeline simulation stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    base_config_defaults = {
        "base_check_time": SIM_CONFIG.get("base_check_time", Config().base_check_time),
        "time_per_occupant": SIM_CONFIG.get("time_per_occupant", Config().time_per_occupant),
        "walk_speed": SIM_CONFIG.get("walk_speed", Config().walk_speed),
        "run_simulation": SIM_CONFIG.get("run_simulation", True),
        "output_root": SIM_CONFIG.get("output_root", args.output_root),
    }
    base_config = Config(
        layout_file=args.layout,
        redundancy_mode=args.redundancy_mode,
        floors=args.floors,
        floor_spacing=args.floor_spacing,
        run_simulation=not args.no_sim,
        output_root=base_config_defaults["output_root"],
        base_check_time=base_config_defaults["base_check_time"],
        time_per_occupant=base_config_defaults["time_per_occupant"],
        walk_speed=base_config_defaults["walk_speed"],
    )
    config = base_config.with_env_overrides()
    _normalize_numeric_config(config)

    layout = load_layout(config.layout_file)
    layout, exits = expand_floors(layout, config.floors, config.floor_spacing)
    problem = EvacuationProblem.from_layout(layout, config, exits)

    logger.info("Global configuration:\n%s", json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    logger.info(
        "Problem stats: floors=%d total_rooms=%d total_occupants=%d responders=%s start_positions=%s",
        config.floors,
        len(list(problem.room_ids)),
        problem.total_occupants,
        list(problem.responder_ids),
        {rid: resp.start_node for rid, resp in problem.responders.items()},
    )

    logger.info("[step 0] Preparing output directory")
    output_dir = prepare_output_dir(
        config.output_root,
        config.layout_file,
        list(problem.responder_ids),
        config.redundancy_mode,
        config.floors,
        problem.total_occupants,
    )
    per_run_log = output_dir / "run.log"
    add_file_handler(per_run_log)
    logger.info("Per-run detailed log: %s", per_run_log)

    logger.info("[step 1/4] Planning sweep for layout=%s floors=%d", Path(config.layout_file).stem, config.floors)
    plan = plan_sweep(problem, config)
    makespan = float(plan.get("makespan", 0.0))
    logger.info("[step 1/4] Planning complete (makespan=%.3f s)", makespan)

    if config.run_simulation:
        logger.info("[step 2/4] Running timeline simulation")
        timeline = simulate_sweep(problem, plan, config)
        event_count = sum(len(entries) for entries in timeline.values())
        logger.info("[step 2/4] Simulation produced %d timeline events", event_count)
    else:
        logger.info("[step 2/4] Simulation skipped (run_simulation=False)")
        timeline = {}

    plan_path = save_plan(output_dir, {"config": config.to_dict(), "plan": plan})
    logger.info("[step 3/4] Saved plan to %s", plan_path)
    timeline_path = None
    if timeline:
        timeline_path = save_timeline(output_dir, timeline)
        logger.info("[step 3/4] Saved timeline to %s", timeline_path)

    summary = summarize_run(problem, plan, timeline, config)
    payload = build_viz_payload(problem, plan, timeline, config)
    logger.info("Room clear order: %s", " -> ".join(summary.get("room_order", [])) or "N/A")
    for rid, order in summary.get("responder_orders", {}).items():
        logger.info("Responder %s search order: %s", rid, " -> ".join(order) or "N/A")
    for rid, info in payload.get("responders", {}).items():
        logger.info("Responder %s path: %s", rid, info.get("path", []))
        timeline_seq = info.get("timeline", [])
        for frame_idx, frame in enumerate(timeline_seq, 1):
            logger.info(
                "[frame %d] %s %s -> %s (type=%s, t0=%.2f, t1=%.2f)",
                frame_idx,
                rid,
                frame.get("from"),
                frame.get("to"),
                frame.get("type"),
                frame.get("t0", 0.0),
                frame.get("t1", 0.0),
            )

    logger.info("[step 4/4] Generating visualization outputs")
    viz_artifacts = export_visuals(problem, config, plan, timeline, output_dir, payload=payload)
    if viz_artifacts:
        logger.info(
            "[step 4/4] Visualization artifacts: %s",
            ", ".join(f"{key}={path.name}" for key, path in viz_artifacts.items()),
        )
    else:
        logger.info("[step 4/4] No visualization artifacts generated")

    summary = summarize_run(problem, plan, timeline, config)
    payload = build_viz_payload(problem, plan, timeline, config)
    logger.info("Room clear order: %s", " -> ".join(summary.get("room_order", [])) or "N/A")
    for rid, order in summary.get("responder_orders", {}).items():
        logger.info("Responder %s search order: %s", rid, " -> ".join(order) or "N/A")
    for rid, info in payload.get("responders", {}).items():
        logger.info("Responder %s path: %s", rid, info.get("path", []))
        timeline_seq = info.get("timeline", [])
        for frame_idx, frame in enumerate(timeline_seq, 1):
            logger.info(
                "[frame %d] %s %s -> %s (type=%s, t0=%.2f, t1=%.2f)",
                frame_idx,
                rid,
                frame.get("from"),
                frame.get("to"),
                frame.get("type"),
                frame.get("t0", 0.0),
                frame.get("t1", 0.0),
            )
    record = build_result_record(problem, plan, timeline, config, output_dir)
    result_csv = Path(config.output_root) / "result.csv"
    append_result_record(record, result_csv)
    logger.info(
        "Sweep summary -> layout=%s floors=%d order=%s makespan=%s",
        record["layout"],
        record["floors"],
        record["order"],
        record["time"],
    )

    print(f"Plan written to {plan_path}")
    if timeline_path:
        print(f"Timeline written to {timeline_path}")
    if viz_artifacts:
        print(f"Visualization assets saved under {output_dir}")
    if record:
        print(f"Run summary appended to {result_csv}")


def _normalize_numeric_config(config: Config) -> None:
    try:
        config.floors = int(config.floors)
    except (TypeError, ValueError):
        config.floors = int(float(config.floors)) if config.floors not in (None, "") else 1
    try:
        config.floor_spacing = float(config.floor_spacing)
    except (TypeError, ValueError):
        config.floor_spacing = float(config.floor_spacing or 60.0)


if __name__ == "__main__":
    main()
