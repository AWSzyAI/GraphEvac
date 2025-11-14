"""New CLI entrypoint that wires configuration, planning and simulation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from graph_evac import (
    Config,
    EvacuationProblem,
    expand_floors,
    load_layout,
    plan_sweep,
    simulate_sweep,
)
from graph_evac.io_utils import prepare_output_dir, save_plan, save_timeline


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
    base_config = Config(
        layout_file=args.layout,
        redundancy_mode=args.redundancy_mode,
        floors=args.floors,
        floor_spacing=args.floor_spacing,
        run_simulation=not args.no_sim,
        output_root=args.output_root,
    )
    config = base_config.with_env_overrides()

    layout = load_layout(config.layout_file)
    layout, exits = expand_floors(layout, config.floors, config.floor_spacing)
    problem = EvacuationProblem.from_layout(layout, config, exits)

    plan = plan_sweep(problem, config)
    timeline = simulate_sweep(problem, plan, config) if config.run_simulation else {}

    output_dir = prepare_output_dir(
        config.output_root,
        config.layout_file,
        list(problem.responder_ids),
        config.redundancy_mode,
        config.floors,
        problem.total_occupants,
    )
    plan_path = save_plan(output_dir, {"config": config.to_dict(), "plan": plan})
    timeline_path = None
    if timeline:
        timeline_path = save_timeline(output_dir, timeline)

    print(f"Plan written to {plan_path}")
    if timeline_path:
        print(f"Timeline written to {timeline_path}")


if __name__ == "__main__":
    main()
