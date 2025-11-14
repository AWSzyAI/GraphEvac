"""Planner facade used by the CLI and tests."""
from __future__ import annotations

from .config import Config
from .problem import EvacuationProblem
from . import greedy


def plan_sweep(problem: EvacuationProblem, config: Config) -> dict:
    """Select a planning backend based on the configuration."""
    # Currently we only expose the greedy approximation but the indirection keeps
    # the public surface stable when we add ILP solvers later.
    return greedy.plan(problem)
