"""
MissionPlanner: responsible for dividing the mission area into balanced search regions,
registering drones, assigning initial search tasks, and handling dynamic handoff when
a drone transitions from searching to tracking a target.
"""

from typing import List, Tuple, Optional, Dict
from math import pi, cos, sin
import numpy as np
from shapely.geometry import Polygon, Point

from mission_state import MissionState
from mission_types import MissionTask, TaskType, TaskStatus
from drone_state import DroneState, Target


class MissionPlanner:
    def __init__(
        self,
        area_bounds: List[Tuple[float, float]],
        grid_size: float,
        home_xy: Optional[Tuple[float, float]] = None,
        desired_alt_m: float = 0.0,
        alt_tolerance_m: float = 2.0,
    ):
        # state holds all global mission bookkeeping
        self.state = MissionState()

        # area_bounds defines the polygon that must be searched
        self.area_bounds = area_bounds

        # grid_size defines the coverage heatmap resolution
        self.grid_size = grid_size

        # home_xy is the return-to-base location (defaults to first drone position if not provided)
        self.home_xy = home_xy

        # altitude constraints are stored on tasks so the planner stays vehicle-agnostic
        self.desired_alt_m = desired_alt_m
        self.alt_tolerance_m = alt_tolerance_m

        # area_poly is created once and reused
        self._area_poly = Polygon(self.area_bounds)

        # task_seq is used to generate unique ids when we later add more task types
        self._task_seq = 0

        # precompute valid coverage cells inside the polygon for fast coverage checks
        self._build_valid_cells()


    def register_drones(self, drones: List[DroneState]) -> None:
        # register drones in the mission state
        for drone in drones:
            self.state.drones[drone.id] = drone

        # if home wasn't set, use the first registered drone as a reasonable default
        if self.home_xy is None and len(drones) > 0:
            self.home_xy = drones[0].position


    def start_mission(self) -> None:
        # create initial search tasks and assign them one-per-drone
        self.create_search_tasks()
        self.assign_initial_search_tasks()


    def _next_task_id(self, prefix: str) -> str:
        # generate unique task ids to avoid collisions across replans
        self._task_seq += 1
        return f"{prefix}_{self._task_seq}"


    def _build_valid_cells(self) -> None:
        # compute which grid cells are inside the mission polygon
        # note: cell indices are derived from polygon bounds and grid_size
        minx, miny, maxx, maxy = self._area_poly.bounds

        # compute integer cell index range spanning the bounds
        # note: using floor-like behavior by int() with division works for positive coords
        # if you expect negative coords, this should be replaced with math.floor
        x0 = int(minx / self.grid_size) - 1
        x1 = int(maxx / self.grid_size) + 1
        y0 = int(miny / self.grid_size) - 1
        y1 = int(maxy / self.grid_size) + 1

        valid: set[Tuple[int, int]] = set()

        # scan cells and keep only those whose center falls inside the polygon
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                center_x = (cx + 0.5) * self.grid_size
                center_y = (cy + 0.5) * self.grid_size
                if self._area_poly.contains(Point(center_x, center_y)):
                    valid.add((cx, cy))

        self.state.valid_cells = valid


    def _cell_index_from_xy(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        # convert an (x, y) position into a grid cell index
        x, y = xy
        return (int(x / self.grid_size), int(y / self.grid_size))


    def ingest_drone_update(
        self,
        drone_id: str,
        position_xy: Tuple[float, float],
        battery: Optional[float] = None,
        is_available: Optional[bool] = None,
    ) -> None:
        # update a drone's state using a status message from the sim/execution layer
        if drone_id not in self.state.drones:
            raise ValueError(f"Unknown drone id: {drone_id}")

        drone = self.state.drones[drone_id]

        # update basic telemetry
        drone.position = position_xy

        if battery is not None:
            drone.battery = battery

        if is_available is not None:
            drone.is_available = is_available

        # update global coverage heatmap based on where the drone is now
        self._update_coverage_with_position(position_xy)


    def _update_coverage_with_position(self, position_xy: Tuple[float, float]) -> None:
        # increment visit count for the grid cell that contains the position
        cell = self._cell_index_from_xy(position_xy)

        # ignore positions outside the mission polygon
        if cell not in self.state.valid_cells:
            return

        # stacking revisits is implemented by storing visit counts per cell
        self.state.visit_counts[cell] = self.state.visit_counts.get(cell, 0) + 1


    def report_target_detection(
        self,
        reporting_drone_id: str,
        target_id: str,
        target_position_xy: Tuple[float, float],
    ) -> None:
        # ingest a target detection from a drone
        # note: v0 assumes the sim gives stable target_ids
        if reporting_drone_id not in self.state.drones:
            raise ValueError(f"Unknown reporting drone id: {reporting_drone_id}")

        # upsert the target track into mission state
        if target_id not in self.state.targets:
            self.state.targets[target_id] = Target(id=target_id, position=target_position_xy)
        else:
            self.state.targets[target_id].position = target_position_xy

        # mark the target as recently seen
        self.state.targets[target_id].last_seen_tick = self.state.tick

        # if nobody is tracking this target yet, assign the reporting drone to track it
        # note: this implements your rule "the drone that spots it becomes the tracker"
        target = self.state.targets[target_id]
        if not target.is_being_tracked:
            self._assign_tracking_task_to_reporting_drone(reporting_drone_id, target_id)


    def tick(self) -> None:
        # advance planner time and run simple rule checks
        self.state.tick += 1

        # run low-battery return-to-base logic as a safety net
        # note: the execution layer can override this later with more realistic constraints
        for drone in self.state.drones.values():
            if drone.battery <= 5.0 and drone.current_task_id is not None:
                # if battery is critically low, force return-to-base
                self._assign_return_to_base(drone.id)


    def _assign_tracking_task_to_reporting_drone(self, drone_id: str, target_id: str) -> None:
        # transition a drone into tracking mode and perform search-task handoff if needed
        drone = self.state.drones[drone_id]

        # if the drone is already tracking something, do nothing (one target per drone)
        if drone.is_tracking:
            return

        # if the drone is currently searching, hand off its search task to someone else
        self._handoff_search_task(from_drone_id=drone_id)

        # create and assign a new tracking task to the reporting drone
        task = MissionTask(
            id=self._next_task_id("track"),
            type=TaskType.TRACK_TARGET,
            target_id=target_id,
            assigned_drone=drone_id,
            status=TaskStatus.ASSIGNED,
            priority=100,
            constraints={
                "desired_alt_m": self.desired_alt_m,
                "alt_tolerance_m": self.alt_tolerance_m,
            },
        )

        # store task in mission state
        self.state.tasks[task.id] = task

        # update drone bookkeeping
        drone.current_task_id = task.id
        drone.is_tracking = True
        drone.is_available = False

        # update target bookkeeping
        target = self.state.targets[target_id]
        target.assigned_drone_id = drone_id
        target.is_being_tracked = True


    def _handoff_search_task(self, from_drone_id: str) -> None:
        # if the drone currently owns a search task, reassign that search task to the nearest available drone
        from_drone = self.state.drones[from_drone_id]
        from_task_id = from_drone.current_task_id

        if from_task_id is None:
            return

        if from_task_id not in self.state.tasks:
            # if task id is missing, clear it to avoid dead references
            from_drone.current_task_id = None
            return

        from_task = self.state.tasks[from_task_id]
        if from_task.type != TaskType.SEARCH_AREA:
            # only search tasks are handed off under mode A
            return

        # unassign the search task from the tracking drone
        from_task.assigned_drone = None
        from_task.status = TaskStatus.UNASSIGNED

        # clear the drone's current task pointer (it will get a tracking task next)
        from_drone.current_task_id = None

        # find a replacement drone that is available and not tracking
        replacement_id = self._find_nearest_available_drone_id(
            origin_xy=from_drone.position,
            exclude_drone_id=from_drone_id,
        )

        if replacement_id is None:
            # if nobody can take it right now, the task stays unassigned
            return

        # assign the task to the replacement drone
        replacement_drone = self.state.drones[replacement_id]
        from_task.assigned_drone = replacement_id
        from_task.status = TaskStatus.ASSIGNED

        replacement_drone.current_task_id = from_task.id
        replacement_drone.is_available = False
        replacement_drone.is_tracking = False


    def _find_nearest_available_drone_id(
        self,
        origin_xy: Tuple[float, float],
        exclude_drone_id: Optional[str] = None,
    ) -> Optional[str]:
        # pick the nearest drone that can accept a new task
        best_id: Optional[str] = None
        best_dist: float = float("inf")

        ox, oy = origin_xy

        for drone in self.state.drones.values():
            # exclude a specific drone if requested (e.g., the tracking drone)
            if exclude_drone_id is not None and drone.id == exclude_drone_id:
                continue

            # require the drone to be idle/available and not tracking
            if not drone.is_available:
                continue
            if drone.is_tracking:
                continue

            dx = drone.position[0] - ox
            dy = drone.position[1] - oy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < best_dist:
                best_dist = dist
                best_id = drone.id

        return best_id


    def _assign_return_to_base(self, drone_id: str) -> None:
        # create a return-to-base task and assign it to the drone
        # note: this preempts any existing task because battery is critical
        if self.home_xy is None:
            raise ValueError("home_xy is not set, cannot return to base")

        drone = self.state.drones[drone_id]

        # mark any current task as failed when we force an emergency rtb
        if drone.current_task_id is not None and drone.current_task_id in self.state.tasks:
            self.state.tasks[drone.current_task_id].status = TaskStatus.FAILED

        task = MissionTask(
            id=self._next_task_id("rtb"),
            type=TaskType.RETURN_TO_BASE,
            assigned_drone=drone_id,
            status=TaskStatus.ASSIGNED,
            priority=1000,
            constraints={
                "home_xy": self.home_xy,
                "desired_alt_m": self.desired_alt_m,
                "alt_tolerance_m": self.alt_tolerance_m,
            },
        )

        self.state.tasks[task.id] = task
        drone.current_task_id = task.id
        drone.is_available = False
        drone.is_tracking = False


    def coverage_fraction(self) -> float:
        # compute fraction of mission cells visited at least once
        total = len(self.state.valid_cells)
        if total == 0:
            return 0.0

        visited = 0
        for cell in self.state.valid_cells:
            if self.state.visit_counts.get(cell, 0) > 0:
                visited += 1

        return visited / total


    def create_search_tasks(self) -> None:
        """
        partition the search area into equal-area wedge sectors based on a reference point.
        each drone receives a single polygonal search region, implemented as one MissionTask.

        note: these search tasks are intended to be stable and handed off (mode A),
        so ids are unique and not reused across missions.
        """

        drones = list(self.state.drones.values())
        num_drones = len(drones)
        if num_drones == 0:
            raise ValueError("No drones registered")

        area_poly = self._area_poly

        # choose a stable reference point for slicing
        # note: we prefer home_xy if provided; otherwise fall back to first drone position
        ref_x, ref_y = self.home_xy if self.home_xy is not None else drones[0].position
        start_point = Point(ref_x, ref_y)

        # compute angle from reference point to center of search area
        area_centroid = area_poly.centroid
        base_angle = np.arctan2(area_centroid.y - start_point.y, area_centroid.x - start_point.x)

        # use full 360-degree spread for generic coverage
        # note: the planner should avoid embedding vehicle assumptions here
        angle_spread = 2 * pi

        # generate evenly spaced angles centered on base direction
        angles = np.linspace(base_angle - angle_spread / 2, base_angle + angle_spread / 2, num_drones + 1)
        target_area = area_poly.area / num_drones

        def make_sector(a1: float, a2: float) -> Polygon:
            # construct triangular sector wedge from reference point
            bounds = area_poly.bounds
            max_x = max(abs(bounds[0] - start_point.x), abs(bounds[2] - start_point.x))
            max_y = max(abs(bounds[1] - start_point.y), abs(bounds[3] - start_point.y))
            radius = np.sqrt(max_x**2 + max_y**2) * 1.5
            p1 = (start_point.x + radius * cos(a1), start_point.y + radius * sin(a1))
            p2 = (start_point.x + radius * cos(a2), start_point.y + radius * sin(a2))
            return Polygon([start_point.coords[0], p1, p2])

        def sector_area(a1: float, a2: float) -> float:
            # compute intersection area between a sector and the mission area
            return make_sector(a1, a2).intersection(area_poly).area

        # iteratively adjust sector boundaries to balance area
        max_iters = 50
        for _ in range(max_iters):
            current_angle = angles[0]
            new_angles = [current_angle]

            for i in range(num_drones):
                a_low = current_angle
                a_high = angles[-1] if i == num_drones - 1 else angles[i + 2]

                for _ in range(20):
                    mid = (a_low + a_high) / 2
                    area = sector_area(current_angle, mid)
                    if area > target_area:
                        a_high = mid
                    else:
                        a_low = mid
                    if abs(area - target_area) < 1.0:
                        break

                new_angles.append(mid)
                current_angle = mid

            if np.allclose(new_angles, angles, atol=1e-3):
                break

            angles = np.array(new_angles)

        # create one search task per drone using clipped angular sectors
        for i in range(num_drones):
            wedge = make_sector(angles[i], angles[i + 1])
            clipped = wedge.intersection(area_poly)

            if clipped.is_empty:
                continue

            task = MissionTask(
                id=self._next_task_id("search"),
                type=TaskType.SEARCH_AREA,
                area=list(clipped.exterior.coords),
                status=TaskStatus.UNASSIGNED,
                priority=10,
                constraints={
                    "desired_alt_m": self.desired_alt_m,
                    "alt_tolerance_m": self.alt_tolerance_m,
                },
            )

            self.state.tasks[task.id] = task


    def assign_initial_search_tasks(self) -> None:
        # assign one search task to each drone that is available
        unassigned_tasks: List[MissionTask] = []
        for task in self.state.tasks.values():
            if task.type == TaskType.SEARCH_AREA and task.status == TaskStatus.UNASSIGNED:
                unassigned_tasks.append(task)

        drone_list = list(self.state.drones.values())

        # sort drones to keep deterministic assignment (useful for tests)
        drone_list.sort(key=lambda d: d.id)

        # sort tasks to keep deterministic assignment (useful for tests)
        unassigned_tasks.sort(key=lambda t: t.id)

        for i, drone in enumerate(drone_list):
            if i >= len(unassigned_tasks):
                break

            # skip drones that aren't available to take a task
            if not drone.is_available:
                continue

            task = unassigned_tasks[i]
            task.assigned_drone = drone.id
            task.status = TaskStatus.ASSIGNED

            drone.current_task_id = task.id
            drone.is_available = False
            drone.is_tracking = False
