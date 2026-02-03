from dataclasses import dataclass
from typing import Tuple, Optional


# represents the state of a single drone in the swarm
@dataclass
class DroneState:
    id: str                                  # unique id of the drone

    # position is 2d xy for the mission planner
    # note: your sim can still be 3d; just keep altitude fixed and store it separately
    position: Tuple[float, float]            # (x, y) coordinates in the environment
    battery: float                           # battery level as a percentage (0–100)

    # altitude_m is a planner-level constraint holder (optional)
    # note: this keeps the planner adaptable to 3d later without breaking 2d logic now
    altitude_m: float = 0.0

    current_task_id: Optional[str] = None    # id of the task currently assigned to the drone
    is_tracking: bool = False                # true if drone is currently tracking a target
    is_available: bool = True                # true if drone can receive a new task


# represents a known or detected target in the environment
@dataclass
class Target:
    id: str                                  # unique identifier for the target
    position: Tuple[float, float]            # (x, y) position of the target
    
    # assigned_drone_id is who is tracking the target, if anyone
    assigned_drone_id: Optional[str] = None
    is_being_tracked: bool = False           # whether a drone is actively tracking it

    # last_seen_tick is used for "lost target" logic
    # note: treat this as a step counter in sim for now
    last_seen_tick: int = 0
