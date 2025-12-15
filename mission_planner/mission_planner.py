"""
MissionPlanner: Responsible for dividing the mission area into balanced search regions,
registering drones, and assigning initial search tasks. Each drone receives one region
to explore based on launch position and area geometry.
"""

from mission_state import MissionState
from mission_types import MissionTask, TaskType, TaskStatus
from drone_state import DroneState
from typing import List, Tuple
from shapely.geometry import Polygon, Point
import numpy as np
from math import pi, cos, sin


class MissionPlanner:

    def __init__(self, area_bounds: List[Tuple[float, float]], grid_size: float):
        """
        Initialize the mission planner.
        - area_bounds: List of (x, y) tuples representing the polygonal mission area.
        - grid_size: Edge length of square search sectors in meters (not used with 1-task-per-drone model).
        """
        self.state = MissionState()
        self.area_bounds = area_bounds
        self.grid_size = grid_size


    def register_drones(self, drones: List[DroneState]):
        """
        Register drones in the mission state. Each drone must have a unique ID and position.
        """
        for drone in drones:
            self.state.drones[drone.id] = drone


    def create_search_tasks(self):
        """
        Partition the search area into equal-area wedge sectors based on the drone launch point.
        Each drone receives a single polygonal search region, implemented as one MissionTask.
        """
        
        # get list of drones, these are DroneState objects
        drones = list(self.state.drones.values())

        num_drones = len(drones)
        if num_drones == 0:
            raise ValueError("No drones registered")

        # convert area bounds to shapely Polygon, and get launch point
        area_poly = Polygon(self.area_bounds)
        launch_x, launch_y = self.state.drones[drones[0].id].position
        start_point = Point(launch_x, launch_y)

        # compute angle from launch point to center of search area
        area_centroid = area_poly.centroid
        base_angle = np.arctan2(area_centroid.y - start_point.y, area_centroid.x - start_point.x)

        # set angular coverage width based on launch distance
        distance_to_area = start_point.distance(area_centroid)
        angle_spread = pi if distance_to_area > 50 else 2 * pi

        # generate evenly spaced angles centered on base direction
        angles = np.linspace(base_angle - angle_spread / 2, base_angle + angle_spread / 2, num_drones + 1)
        target_area = area_poly.area / num_drones

        def make_sector(a1, a2):
            # construct triangular sector wedge from launch point
            bounds = area_poly.bounds
            max_x = max(abs(bounds[0] - start_point.x), abs(bounds[2] - start_point.x))
            max_y = max(abs(bounds[1] - start_point.y), abs(bounds[3] - start_point.y))
            radius = np.sqrt(max_x**2 + max_y**2) * 1.5
            p1 = (start_point.x + radius * cos(a1), start_point.y + radius * sin(a1))
            p2 = (start_point.x + radius * cos(a2), start_point.y + radius * sin(a2))
            return Polygon([start_point.coords[0], p1, p2])

        def sector_area(a1, a2):
            # compute the intersection area between a sector and the mission area
            return make_sector(a1, a2).intersection(area_poly).area

        # iteratively adjust sector boundaries to balance area
        max_iters = 50
        for iteration in range(max_iters):
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

        # create one task per drone using clipped angular sectors
        for i in range(num_drones):
            wedge = make_sector(angles[i], angles[i + 1])
            clipped = wedge.intersection(area_poly)

            if clipped.is_empty:
                continue

            task = MissionTask(
                id=f"search_{i}",
                type=TaskType.SEARCH_AREA,
                area=list(clipped.exterior.coords)
            )
            self.state.tasks[task.id] = task


    def assign_initial_search_tasks(self):
        """
        Assign one available search task to each drone that was registered.
        This assumes that create_search_tasks() has already been called,
        and that one search task exists per drone.
        """
        # initialize an empty list to store tasks that are unassigned and of type SEARCH_AREA
        unassigned_tasks = []

        # loop through all tasks stored in the mission state
        for task in self.state.tasks.values():
            # we're only interested in tasks that haven't been assigned yet
            # and are specifically SEARCH_AREA tasks (not TRACK_TARGET or other types)
            if task.status == TaskStatus.UNASSIGNED and task.type == TaskType.SEARCH_AREA:
                # add the valid task to the list of unassigned search tasks
                unassigned_tasks.append(task)

        # retrieve a list of all drones currently registered in the mission state
        drone_list = list(self.state.drones.values())

        # loop through all drones using their index and object
        for i, drone in enumerate(drone_list):
            # if there are fewer tasks than drones, stop assigning once we run out of tasks
            if i >= len(unassigned_tasks):
                break

            # get the i-th unassigned task from the filtered task list
            task = unassigned_tasks[i]

            # assign the task to the current drone by setting the drone's ID in the task
            task.assigned_drone = drone.id

            # update the task status to reflect that it's now assigned
            task.status = TaskStatus.ASSIGNED

            # record in the drone's state which task it's now responsible for
            drone.current_task_id = task.id
