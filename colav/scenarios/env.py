from colav.path.pwl import PWLPath
from colav.obstacles.moving import MovingObstacle, MovingShip
from colav.planner import TimeSpaceColav
from colav.path.planning import PathPlanner
from shapely import Polygon
from typing import List, Optional, Tuple, Dict, Any
from math import atan2, pi

class COLAVEnv:
    def __init__(
        self,
        own_ship: MovingShip,
        path: PWLPath,
        desired_speed: float,
        obstacles: Optional[List[MovingShip]] = None,
        distance_threshold: float = 3e3, # Distance at which colav is enabled
        shore: Optional[ List[Polygon] ] = None,
        max_speed: float = float('inf'),
        max_course_rate: float = float('inf'),
        colregs: bool = False,
        good_seamanship: bool = False,
        path_planner: Optional[PathPlanner] = None,
        max_iter: int = 10, # Max iterations for finding a valid solution
        speed_factor: float = 0.95, # At each iteration k, speed = speed_factor**k * desired_speed
        degrees: bool = True,
        lookahead_distance: float = 300,
        buffer_moving: float = 0,
        simplify_moving: float = 0,
        buffer_static: float = 0,
        simplify_static: float = 0,
        abort_colregs_after_iter: Optional[int] = None
    ):
        self.own_ship = own_ship
        self.path = path
        self.desired_speed = desired_speed
        self.shore: List[Polygon] = shore or []
        self.obstacles = obstacles or []
        self.lookahead_distance = lookahead_distance
        self.buffer_moving = buffer_moving
        self.simplify_moving = simplify_moving
        self.buffer_static = buffer_static
        self.simplify_static = simplify_static

        self.planner = TimeSpaceColav(
            desired_speed=desired_speed,
            distance_threshold=distance_threshold,
            shore=[Polygon(obs.buffer(self.buffer_static).simplify(self.simplify_static).boundary.coords) for obs in self.shore],
            max_speed=max_speed,
            max_course_rate=max_course_rate, 
            colregs=colregs,
            good_seamanship=good_seamanship,
            path_planner=path_planner,
            max_iter=max_iter,
            speed_factor=speed_factor,
            degrees=degrees,
            abort_colregs_after_iter=abort_colregs_after_iter
        )

    def step(self, dt: float) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        progression = self.path.progression(*self.own_ship.position)    # Own ship's progression along global path
        
        # Compute collision-free trajectory
        traj, info = self.planner.get(
            self.own_ship.position,
            self.path.interpolate(progression + self.lookahead_distance),
            [obs.buffer(self.buffer_moving) for obs in self.obstacles],
            heading=self.own_ship.psi,
            degrees=self.own_ship.degrees
        )

        if traj is not None:
            speed, heading = traj.get_speed(0), traj.get_heading(0, degrees=self.own_ship.degrees)
        else:
            speed, heading = 0, self.own_ship.psi

        # Limit course rate to feasible values
        desired_course_rate = (heading - self.own_ship.psi) / dt
        # take double of the planner max course rate to have some margin
        course_rate = max(min(desired_course_rate, 2 * self.planner.max_course_rate), - 2 * self.planner.max_course_rate)
        
        # Limit acceleration
        desired_acceleration = (speed - self.own_ship.u) / dt
        acc = max(min(desired_acceleration, 0.02), -0.02)

        # Integrate
        self.own_ship = MovingShip.from_body(self.own_ship.position, self.own_ship.psi + course_rate * dt, self.own_ship.u + acc * dt, 0, self.own_ship.loa, self.own_ship.beam, degrees=self.own_ship.degrees, mmsi=self.own_ship.mmsi)

        self.own_ship = self.own_ship.predict(dt)
        for i in range(len(self.obstacles)):
            self.obstacles[i] = self.obstacles[i].resample_velocity().predict(dt)

        return {'own_ship': self.own_ship, 'obstacles': self.obstacles, 'trajectory': traj}, self.path.progression(*self.own_ship.position) >= self.path.length, info