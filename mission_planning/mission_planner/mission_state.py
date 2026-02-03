from typing import Dict, Tuple, Set
from .mission_types import MissionTask
from .drone_state import DroneState, Target
from dataclasses import dataclass, field


# global mission state that holds all tracked entities during execution
@dataclass
class MissionState:
    # dictionary mapping task id to its corresponding MissionTask
    tasks: Dict[str, MissionTask] = field(default_factory=dict)

    # dictionary mapping drone id to its current DroneState
    drones: Dict[str, DroneState] = field(default_factory=dict)

    # dictionary mapping target id to its last known state
    targets: Dict[str, Target] = field(default_factory=dict)

    # visit_counts is the global coverage heatmap (stacking revisits)
    # key: (cell_x, cell_y) index in the grid, value: number of times any drone visited it
    visit_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # valid_cells is the set of grid cells that are inside the mission polygon
    # note: this lets you compute coverage percent without scanning empty space
    valid_cells: Set[Tuple[int, int]] = field(default_factory=set)

    # tick is a simple time step counter the planner can increment each update
    tick: int = 0
