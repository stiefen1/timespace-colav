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
        obstacles: Optional[List[MovingObstacle]] = None,
        distance_threshold: float = 3e3, # Distance at which colav is enabled
        shore: Optional[ List[Polygon] ] = None,
        max_speed: float = float('inf'),
        max_yaw_rate: float = float('inf'),
        colregs: bool = False,
        good_seamanship: bool = False,
        path_planner: Optional[PathPlanner] = None,
        max_iter: int = 10, # Max iterations for finding a valid solution
        speed_factor: float = 0.95, # At each iteration k, speed = speed_factor**k * desired_speed
        degrees: bool = True,
        lookahead_distance: float = 300,
        buffer: float = 0,
        simplify: float = 0
    ):
        self.own_ship = own_ship
        self.path = path
        self.desired_speed = desired_speed
        self.shore: List[Polygon] = shore or []
        self.obstacles = obstacles or []
        self.lookahead_distance = lookahead_distance
        self.buffer = buffer
        self.simplify = simplify

        self.planner = TimeSpaceColav(
            desired_speed=desired_speed,
            distance_threshold=distance_threshold,
            shore=shore,
            max_speed=max_speed,
            max_yaw_rate=max_yaw_rate,
            colregs=colregs,
            good_seamanship=good_seamanship,
            path_planner=path_planner,
            max_iter=max_iter,
            speed_factor=speed_factor,
            degrees=degrees
        )

    def step(self, dt: float) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        progression = self.path.progression(*self.own_ship.position)
        
        # print("p0: ", p0, "pf: ", pf, "angle: ", 180*atan2(pf[0]-p0[0], pf[1]-p0[1])/pi)
        traj, info = self.planner.get(
            self.own_ship.position,
            [obs.buffer(self.buffer).simplify(self.simplify) for obs in self.obstacles],
            pf=self.path.interpolate(progression + self.lookahead_distance),
            heading=self.own_ship.psi,
            degrees=self.own_ship.degrees
        )

        if traj is not None:
            speed, heading = traj.get_speed(0), traj.get_heading(0, degrees=self.own_ship.degrees)
            self.own_ship = MovingShip.from_body(self.own_ship.position, heading, speed, 0, self.own_ship.loa, self.own_ship.beam, degrees=self.own_ship.degrees, mmsi=self.own_ship.mmsi)

        self.own_ship = self.own_ship.predict(dt)
        self.obstacles = [ts.predict(dt) for ts in self.obstacles]        

        return {'own_ship': self.own_ship, 'obstacles': self.obstacles}, self.path.progression(*self.own_ship.position) >= self.path.length, info