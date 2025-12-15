from typing import Dict
from mission_types import MissionTask
from drone_state import DroneState, Target
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
