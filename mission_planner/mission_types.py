from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, List, Optional, Dict, Any


# types of tasks a drone might perform
class TaskType(Enum):
    SEARCH_AREA = auto()       # search a region
    TRACK_TARGET = auto()      # track a known or detected target
    RETURN_TO_BASE = auto()    # return to home/launch location


# track the status of a task during the mission
class TaskStatus(Enum):
    UNASSIGNED = auto()        # task hasn't been assigned to a drone
    ASSIGNED = auto()          # task is assigned but not started
    IN_PROGRESS = auto()       # task is currently being executed
    COMPLETED = auto()         # task finished successfully
    FAILED = auto()            # drone was unable to complete the task


# represents a task assigned to a drone, such as search or tracking
@dataclass
class MissionTask:
    
    # unique task id
    id: str
    
    # type of task being performed
    type: TaskType

    # area is a polygon for search tasks
    # the mission planner stays 2d for now (xy), even if the sim is 3d
    area: Optional[List[Tuple[float, float]]] = None

    # target_id is used for tracking tasks
    target_id: Optional[str] = None

    # assigned_drone is which drone currently owns this task
    assigned_drone: Optional[str] = None

    # status describes task lifecycle
    status: TaskStatus = TaskStatus.UNASSIGNED

    # dependencies lets you express ordering later (not required for v1)
    dependencies: List[str] = field(default_factory=list)

    # priority lets the planner choose among competing tasks (higher = more important)
    priority: int = 0

    # constraints is where "vehicle-agnostic" planning lives
    # example: {"desired_alt_m": 30.0, "alt_tolerance_m": 2.0, "time_budget_s": 120}
    constraints: Dict[str, Any] = field(default_factory=dict)
