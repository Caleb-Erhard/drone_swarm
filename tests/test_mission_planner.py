from mission_planner.mission_planner import MissionPlanner
from mission_planner.drone_state import DroneState
from mission_planner.mission_types import TaskType, TaskStatus
import pytest


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

def test_low_battery_triggers_return_to_base():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0, battery=10.0)
    planner.register_drones([d1])
    planner.start_mission()
    
    # Simulate battery drain
    planner.state.drones["drone_1"].battery = 4.0
    planner.tick()
    
    task_id = planner.state.drones["drone_1"].current_task_id
    assert task_id is not None
    assert planner.state.tasks[task_id].type == TaskType.RETURN_TO_BASE

def test_return_to_base_marks_previous_task_as_failed():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0, battery=100.0)
    planner.register_drones([d1])
    planner.start_mission()
    
    original_task_id = planner.state.drones["drone_1"].current_task_id
    planner.state.drones["drone_1"].battery = 4.0
    planner.tick()
    
    assert planner.state.tasks[original_task_id].status == TaskStatus.FAILED

def test_register_drones_with_no_home_xy_uses_first_drone_position():
    # Tests the fallback behavior when home_xy is None
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0)
    d1 = _make_drone("drone_1", 50.0, 50.0)
    planner.register_drones([d1])
    assert planner.home_xy == (50.0, 50.0)

def test_ingest_drone_update_with_unknown_drone_raises_error():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    with pytest.raises(ValueError, match="Unknown drone id"):
        planner.ingest_drone_update("unknown_drone", (10.0, 10.0))

def test_report_target_detection_with_unknown_drone_raises_error():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    with pytest.raises(ValueError, match="Unknown reporting drone id"):
        planner.report_target_detection("unknown_drone", "target_1", (50.0, 50.0))

def test_create_search_tasks_with_no_drones_raises_error():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    with pytest.raises(ValueError, match="No drones registered"):
        planner.create_search_tasks()

def test_coverage_fraction_starts_at_zero():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    assert planner.coverage_fraction() == 0.0

def test_coverage_fraction_increases_with_drone_movement():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    planner.register_drones([d1])
    
    # Move drone to different cells
    planner.ingest_drone_update("drone_1", (10.0, 10.0))
    coverage_1 = planner.coverage_fraction()
    
    planner.ingest_drone_update("drone_1", (30.0, 30.0))
    coverage_2 = planner.coverage_fraction()
    
    assert coverage_2 > coverage_1

def test_coverage_counts_revisits():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    planner.register_drones([d1])
    
    planner.ingest_drone_update("drone_1", (10.0, 10.0))
    planner.ingest_drone_update("drone_1", (10.0, 10.0))
    
    cell = planner._cell_index_from_xy((10.0, 10.0))
    assert planner.state.visit_counts[cell] == 2

def test_unavailable_drone_is_not_assigned_tasks():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0, available=False)
    d2 = _make_drone("drone_2", 12.0, 12.0, available=True)
    
    planner.register_drones([d1, d2])
    planner.start_mission()
    
    assert planner.state.drones["drone_1"].current_task_id is None
    assert planner.state.drones["drone_2"].current_task_id is not None

def test_drone_already_tracking_is_not_reassigned():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    planner.register_drones([d1])
    planner.start_mission()
    
    planner.state.drones["drone_1"].is_available = True
    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))
    
    # Try to detect another target - should not reassign
    original_task = planner.state.drones["drone_1"].current_task_id
    planner.report_target_detection("drone_1", "target_2", (60.0, 60.0))
    
    assert planner.state.drones["drone_1"].current_task_id == original_task

def test_multiple_drones_can_track_different_targets():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    d2 = _make_drone("drone_2", 12.0, 12.0)
    
    planner.register_drones([d1, d2])
    planner.start_mission()
    
    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))
    planner.report_target_detection("drone_2", "target_2", (60.0, 60.0))
    
    assert planner.state.drones["drone_1"].is_tracking
    assert planner.state.drones["drone_2"].is_tracking
    assert planner.state.targets["target_1"].assigned_drone_id == "drone_1"
    assert planner.state.targets["target_2"].assigned_drone_id == "drone_2"

def test_task_handoff_when_no_available_drones_leaves_task_unassigned():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0, available=True)
    d2 = _make_drone("drone_2", 12.0, 12.0, available=False)
    
    planner.register_drones([d1, d2])
    planner.start_mission()
    
    d1_search_task = planner.state.drones["drone_1"].current_task_id
    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))
    
    # Original search task should be unassigned (no one to take it)
    assert planner.state.tasks[d1_search_task].status == TaskStatus.UNASSIGNED
    assert planner.state.tasks[d1_search_task].assigned_drone is None

def test_target_updates_position_on_repeated_detection():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    planner.register_drones([d1])
    
    planner.report_target_detection("drone_1", "target_1", (50.0, 50.0))
    assert planner.state.targets["target_1"].position == (50.0, 50.0)
    
    planner.report_target_detection("drone_1", "target_1", (55.0, 55.0))
    assert planner.state.targets["target_1"].position == (55.0, 55.0)

def test_track_failed_with_unknown_target_still_returns_drone_to_search():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0, available=True)
    d2 = _make_drone("drone_2", 12.0, 12.0, available=True)
    
    planner.register_drones([d1, d2])
    planner.start_mission()
    
    # Free up drone_2's task
    d2_task = planner.state.drones["drone_2"].current_task_id
    planner.state.tasks[d2_task].status = TaskStatus.UNASSIGNED
    planner.state.tasks[d2_task].assigned_drone = None
    planner.state.drones["drone_2"].current_task_id = None
    
    # Report failure on unknown target
    planner.report_track_failed("drone_1", "unknown_target", (50.0, 50.0))
    
    # Drone should still return to search
    assert planner.state.drones["drone_1"].is_tracking is False
    assert planner.state.drones["drone_1"].current_task_id is not None

def test_pop_pending_assignments_returns_and_clears_assignments():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 10.0, 10.0)
    planner.register_drones([d1])
    planner.start_mission()
    
    assignments = planner.pop_pending_assignments()
    assert "drone_1" in assignments
    
    # Second call should be empty
    assignments_2 = planner.pop_pending_assignments()
    assert len(assignments_2) == 0

def test_tick_increments_counter():
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    assert planner.state.tick == 0
    planner.tick()
    assert planner.state.tick == 1
    planner.tick()
    assert planner.state.tick == 2

def test_coverage_tracking_works_correctly():
    """Test that the planner correctly tracks coverage as drones report positions"""
    planner = MissionPlanner(area_bounds=_make_square_area(), grid_size=10.0, home_xy=(0.0, 0.0))
    d1 = _make_drone("drone_1", 0.0, 0.0)
    planner.register_drones([d1])
    
    # Initially no coverage
    assert planner.coverage_fraction() == 0.0
    
    # Visit one cell
    planner.ingest_drone_update("drone_1", (5.0, 5.0))
    coverage_1 = planner.coverage_fraction()
    assert coverage_1 > 0.0
    
    # Visit a different cell - coverage should increase
    planner.ingest_drone_update("drone_1", (25.0, 25.0))
    coverage_2 = planner.coverage_fraction()
    assert coverage_2 > coverage_1
    
    # Visit same cell again - coverage shouldn't increase
    planner.ingest_drone_update("drone_1", (25.0, 25.0))
    coverage_3 = planner.coverage_fraction()
    assert coverage_3 == coverage_2
    
    # But visit count should increase
    cell = planner._cell_index_from_xy((25.0, 25.0))
    assert planner.state.visit_counts[cell] == 2
