"""Configuration utilities for the GraphEvac simulations."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import os
from pathlib import Path
from typing import Any, Dict


def _coerce_value(expected_type: type, value: str) -> Any:
    """Coerce a string value coming from the environment into ``expected_type``."""
    if expected_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if expected_type is int:
        return int(value)
    if expected_type is float:
        return float(value)
    if expected_type is Path:
        return Path(value)
    return value


@dataclass
class Config:
    """Container for simulation and planning parameters."""

    layout_file: str = os.path.join("layout", "baseline.json")
    redundancy_mode: str = "assignment"
    empirical_mode: str | None = None
    base_check_time: float = 5.0
    time_per_occupant: float = 3.0
    walk_speed: float = 1.0
    floors: int = 1
    floor_spacing: float = 20.0
    run_simulation: bool = True
    output_root: str = "out"

    def with_env_overrides(self) -> "Config":
        """Return a copy of the config with environment overrides applied."""
        data: Dict[str, Any] = asdict(self)
        for field in fields(self):
            env_key = field.name.upper()
            if env_key in os.environ:
                raw_value = os.environ[env_key]
                data[field.name] = _coerce_value(field.type, raw_value)
        return Config(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
