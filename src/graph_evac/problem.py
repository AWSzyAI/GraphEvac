"""Domain data structures used by the planners."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .config import Config
from .layout import Layout

Coord = Tuple[float, float]


@dataclass(frozen=True)
class Room:
    room_id: str
    coord: Coord
    occupants: int


@dataclass(frozen=True)
class Responder:
    responder_id: str
    start_node: str


@dataclass
class EvacuationProblem:
    rooms: Dict[str, Room]
    responders: Dict[str, Responder]
    coords: Dict[str, Coord]
    exits: List[str]
    config: Config

    @classmethod
    def from_layout(cls, layout: Layout, config: Config, exits: List[str]) -> "EvacuationProblem":
        rooms = {
            rid: Room(rid, layout.coords.get(rid, (0.0, 0.0)), int(layout.occupants.get(rid, 0)))
            for rid in layout.rooms
        }
        responders = {
            rid: Responder(rid, layout.start_nodes.get(rid, rid)) for rid in layout.responders
        }
        coords = dict(layout.coords)
        coords.update({room.room_id: room.coord for room in rooms.values()})
        return cls(rooms, responders, coords, exits, config)

    def node_coord(self, node: str) -> Coord:
        coord = self.coords.get(node)
        if coord is not None:
            return (float(coord[0]), float(coord[1]))
        if node in self.rooms:
            room = self.rooms[node]
            return room.coord
        return (0.0, 0.0)

    def distance(self, node_a: str, node_b: str) -> float:
        ax, ay = self.node_coord(node_a)
        bx, by = self.node_coord(node_b)
        dx = ax - bx
        dy = ay - by
        return (dx * dx + dy * dy) ** 0.5

    def room_service_time(self, room_id: str) -> float:
        room = self.rooms[room_id]
        return self.config.base_check_time + room.occupants * self.config.time_per_occupant

    @property
    def room_ids(self) -> Iterable[str]:
        return self.rooms.keys()

    @property
    def responder_ids(self) -> Iterable[str]:
        return self.responders.keys()

    @property
    def total_occupants(self) -> int:
        return sum(room.occupants for room in self.rooms.values())
