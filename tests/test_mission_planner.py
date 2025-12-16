from mission_planner.mission_planner import MissionPlanner
from mission_planner.drone_state import DroneState
from mission_planner.mission_types import TaskType, TaskStatus


def _make_square_area(size: float = 100.0):
    return [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]


def _make_drone(drone_id: str, x: float, y: float, battery: float = 100.0, available: bool = True):
    return DroneState(
        id=drone_id,
        position=(x, y),
        battery=battery,
        is_available=available,
    )


def test_initial_search_assignment_assigns_one_search_task_per_available_drone():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(-10.0, -10.0))

    d1 = _make_drone("drone_1", -10.0, -10.0, available=True)
    d2 = _make_drone("drone_2", -12.0, -8.0, available=True)

    planner.register_drones([d1, d2])
    planner.start_mission()

    search_tasks = [t for t in planner.state.tasks.values() if t.type == TaskType.SEARCH_AREA]
    assert len(search_tasks) > 0

    t1_id = planner.state.drones["drone_1"].current_task_id
    t2_id = planner.state.drones["drone_2"].current_task_id

    assert t1_id is not None
    assert t2_id is not None
    assert t1_id != t2_id

    t1 = planner.state.tasks[t1_id]
    t2 = planner.state.tasks[t2_id]

    assert t1.type == TaskType.SEARCH_AREA
    assert t2.type == TaskType.SEARCH_AREA
    assert t1.assigned_drone == "drone_1"
    assert t2.assigned_drone == "drone_2"
    assert t1.status == TaskStatus.ASSIGNED
    assert t2.status == TaskStatus.ASSIGNED


def test_target_detection_assigns_tracking_and_handoffs_search_task():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(-10.0, -10.0))

    d1 = _make_drone("drone_1", -10.0, -10.0, available=True)
    d2 = _make_drone("drone_2", -12.0, -8.0, available=True)

    planner.register_drones([d1, d2])
    planner.start_mission()

    search_tasks = [t for t in planner.state.tasks.values() if t.type == TaskType.SEARCH_AREA]
    assert len(search_tasks) > 0

    d1_search_task_id = planner.state.drones["drone_1"].current_task_id
    assert d1_search_task_id is not None
    assert planner.state.tasks[d1_search_task_id].type == TaskType.SEARCH_AREA

    # simulate execution layer: drone_2 is now idle and can accept a handoff
    planner.state.drones["drone_2"].current_task_id = None
    planner.state.drones["drone_2"].is_available = True

    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))

    d1_task_id = planner.state.drones["drone_1"].current_task_id
    assert d1_task_id is not None
    assert planner.state.tasks[d1_task_id].type == TaskType.TRACK_TARGET
    assert planner.state.tasks[d1_task_id].target_id == "target_1"
    assert planner.state.drones["drone_1"].is_tracking is True

    d2_task_id = planner.state.drones["drone_2"].current_task_id
    assert d2_task_id == d1_search_task_id
    assert planner.state.tasks[d2_task_id].assigned_drone == "drone_2"


def test_track_failed_releases_target_and_returns_drone_to_search_if_possible():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(-10.0, -10.0))

    d1 = _make_drone("drone_1", -10.0, -10.0, available=True)
    d2 = _make_drone("drone_2", -12.0, -8.0, available=True)

    planner.register_drones([d1, d2])
    planner.start_mission()

    search_tasks = [t for t in planner.state.tasks.values() if t.type == TaskType.SEARCH_AREA]
    assert len(search_tasks) > 0

    # free drone_2 to take the handoff task
    planner.state.drones["drone_2"].current_task_id = None
    planner.state.drones["drone_2"].is_available = True

    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))
    assert planner.state.drones["drone_1"].is_tracking is True

    # ensure drone_1 can be re-assigned to search
    planner.state.drones["drone_1"].is_available = True

    # create at least one unassigned search task by freeing drone_2's current search task (if any)
    d2_task_id = planner.state.drones["drone_2"].current_task_id
    if d2_task_id is not None:
        planner.state.tasks[d2_task_id].assigned_drone = None
        planner.state.tasks[d2_task_id].status = TaskStatus.UNASSIGNED
        planner.state.drones["drone_2"].current_task_id = None

    planner.report_track_failed("drone_1", "target_1", (52.0, 52.0))

    assert planner.state.drones["drone_1"].is_tracking is False
    assert planner.state.targets["target_1"].is_being_tracked is False
    assert planner.state.targets["target_1"].assigned_drone_id is None

    new_task_id = planner.state.drones["drone_1"].current_task_id
    assert new_task_id is not None
    assert planner.state.tasks[new_task_id].type == TaskType.SEARCH_AREA
    assert planner.state.tasks[new_task_id].assigned_drone == "drone_1"
