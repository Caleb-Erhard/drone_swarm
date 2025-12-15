from dataclasses import dataclass
from typing import Tuple, Optional

# represents the state of a single drone in the swarm
@dataclass
class DroneState:
    id: str                                # unique id of the drone
    position: Tuple[float, float]          # (x, y) coordinates in the environment
    battery: float                         # battery level as a percentage (0–100)
    current_task_id: Optional[str] = None  # id of the task currently assigned to the drone
    is_tracking: bool = False              # true if drone is currently tracking a target
    is_available: bool = True              # true if drone is idle and can receive new tasks

# represents a known or detected target in the environment
@dataclass
class Target:
    id: str                                  # unique identifier for the target
    position: Tuple[float, float]            # (x, y) position of the target
    assigned_drone_id: Optional[str] = None  # id of drone tracking this target, if any
    is_being_tracked: bool = False           # whether a drone is actively tracking it
