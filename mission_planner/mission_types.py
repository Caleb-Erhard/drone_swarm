from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, List, Optional, Dict

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
    id: str                                                # unique task id
    type: TaskType                                         # type of task being performed
    area: Optional[List[Tuple[float, float]]] = None       # polygonal region (used for SEARCH_AREA tasks)
    target_id: Optional[str] = None                        # for tracking: id of the target to follow
    assigned_drone: Optional[str] = None                   # id of the drone assigned to this task
    status: TaskStatus = TaskStatus.UNASSIGNED             # current status of this task
    dependencies: List[str] = field(default_factory=list)  # tasks that must be completed before this one
    priority: int = 0                                      # priority for scheduling (higher = more important)
