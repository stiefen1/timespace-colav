
"""
Typical use case:

"""

from typing import List, Tuple, Optional, Dict
from colav.obstacles import MovingObstacle, MovingShip
from colav.timespace.projector import TimeSpaceProjector
from colav.path.planning import PathPlanner
from colav.path.pwl import PWLPath, PWLTrajectory
from colav.path.planning import VGPathPlanner
from colav.timespace.constraints import SpeedConstraint, YawRateConstraint, COLREGS
import shapely, logging
from colav.utils.math import DEG2RAD, RAD2DEG, ssa
from math import atan2, cos, sin
logger = logging.getLogger(__name__)

class TimeSpaceColav:
    def __init__(
            self,
            desired_speed: float,
            distance_threshold: float = 3e3, # Distance at which colav is enabled
            shore: Optional[ List[shapely.Polygon] ] = None,
            max_speed: float = float('inf'),
            max_course_rate: float = float('inf'),
            colregs: bool = False,
            good_seamanship: bool = False,
            path_planner: Optional[PathPlanner] = None,
            max_iter: int = 10, # Max iterations for finding a valid solution
            speed_factor: float = 0.95, # At each iteration k, speed = speed_factor**k * desired_speed
            degrees: bool = True,
            abort_colregs_after_iter: Optional[int] = None
    ):
        assert desired_speed > 0, f"Desired speed must be > 0. Got {desired_speed} <= 0 instead."
        assert max_speed > 0, f"Maximum speed must be > 0. Got {max_speed} <= 0 instead."
        assert max_course_rate > 0, f"Maximum course rate must be > 0. Got {max_course_rate} <= 0 instead."
        assert 0 < speed_factor <= 1, f"Speed factor must be in ]0, 1]. Got {speed_factor} instead."

        self.desired_speed = desired_speed
        self.distance_threshold = distance_threshold
        self.shore = shore or []
        self.max_speed = max_speed
        self.max_course_rate = max_course_rate
        self.colregs = colregs
        self.projector = TimeSpaceProjector(self.desired_speed)
        self.path_planner = path_planner or PathPlanner()
        self.max_iter = max_iter
        self.speed_factor = speed_factor
        self.abort_colregs_after_iter = abort_colregs_after_iter or max_iter // 2 # If abort_colregs_after_iter is not specified, we abort it at max_iter // 2
        self.edge_filters = [
            SpeedConstraint(speed=max_speed),
            YawRateConstraint(course_rate=DEG2RAD(max_course_rate) if degrees else max_course_rate)
        ]
        self.node_filters = [COLREGS(good_seamanship=good_seamanship)] if colregs else []

        logger.debug(f"Successfully initialized TimeSpaceColav")

    def get(
            self, 
            p0: Tuple[float, float], 
            obstacles: List[MovingShip], 
            *args, 
            heading: Optional[float] = None, 
            desired_heading: Optional[float] = None,
            pf: Optional[Tuple[float, float]] = None, 
            margin: float = 0.0, 
            degrees: bool = True,
            lookahead_distance: float = 300, # distance of the artificial pf from p0. Only used if pf is not provided.
            ts_in_TSS: bool = False,
            os_in_TSS: bool = False,
            good_seamanship: bool = False,
            **kwargs
        ) -> Tuple[Optional[PWLTrajectory], Dict]:
        """
        Returns the shortest trajectory between p0 and pf to avoid obstacles as a piecewise-linear path parameterized in time.

        If heading is provided and self.max_course_rate was provided at initialization, it can be used as a constraint. 

        margin is the dilatation applied to the obstacles through the shapely.buffer() method.

        Keywords argument can be used to affect the shapely.buffer method's behaviour
        """
        # TODO: Add desired_heading: Optional[float] = None as a replacement to pf

        assert desired_heading is not None or pf is not None, f"Either desired_heading or pf must be provided. Got desired_heading={desired_heading} and pf={pf}"

        if pf is not None: # pf has priority over desired_heading
            desired_heading = atan2(pf[0]-p0[0], pf[1]-p0[1])
        elif desired_heading is not None:
            desired_heading = DEG2RAD(desired_heading) if degrees else desired_heading
            pf = (p0[0] + lookahead_distance * sin(desired_heading), p0[1] + lookahead_distance * cos(desired_heading))

        if (heading is not None) and (desired_heading is not None):
            heading = DEG2RAD(heading) if degrees else heading

            # Raise warning if heading is very different from the actual direction leading to pf
            angle_error = ssa(heading - desired_heading)
            if abs(angle_error) >= DEG2RAD(90):
                logger.warning(f"Heading angle is {RAD2DEG(heading):.0f} degrees, but target point is oriented towards {RAD2DEG(desired_heading):.0f} degrees")

        buffered_obstacles: List[MovingShip] = []
        for obs in obstacles:
            if obs.distance(*p0) <= self.distance_threshold:
                buffered_obstacles.append(obs.buffer(margin, **kwargs) if margin > 0 else obs)

        discount_power = 0
        colregs_active = self.colregs
        for k in range(self.max_iter):
            # Decrease desired speed at each iteration to find a plane that admits at least one feasible path
            # self.projector.v_des = (self.speed_factor**discount_power) * self.desired_speed
            self.projector.v_des = (1 / (1 + 5 * discount_power / self.max_iter)) * self.desired_speed
            logger.info(f"iteration {k+1}/{self.max_iter} | minimum speed = {self.projector.v_des:.1f} (discount power = {discount_power})")
            discount_power += 1

            # Get timespace footprint, i.e. static polygons to be avoided
            projected_obstacles: List[shapely.Polygon] = self.projector.get(
                p0,
                pf,
                buffered_obstacles
            )

            # Convert projected (moving) obstacles and shore into dict
            projected_obstacles_as_dict = {obs.mmsi: proj_obs for obs, proj_obs in zip(buffered_obstacles, projected_obstacles)} 
            moving_obstacles_as_dict = {obs.mmsi: obs for obs in buffered_obstacles}
            shore_as_dict = {i+1: self.shore[i] for i in range(len(self.shore))}

            # Disable colregs if no solution was found before
            if k >= self.abort_colregs_after_iter:
                colregs_active = False

            # Active node filters
            active_node_filters = []
            for node_filter in self.node_filters:
                if isinstance(node_filter, COLREGS):
                    if colregs_active:
                        active_node_filters.append(node_filter)
                    else:
                        if k == self.abort_colregs_after_iter:
                            discount_power = 0
                            logger.warning(f"Iteration {k+1}: aborting COLREGS compliance")
                else:
                    active_node_filters.append(node_filter)

            # Create path planner
            self.path_planner = VGPathPlanner(
                p0,
                pf,
                obstacles = projected_obstacles_as_dict | shore_as_dict,
                edge_filters = self.edge_filters,
                node_filters = active_node_filters,
                plane = self.projector.plane,
                heading = heading,
                moving_obstacles_as_dict = moving_obstacles_as_dict, # Useful for COLREGs
                degrees = degrees,
                ts_in_TSS = ts_in_TSS,
                os_in_TSS = os_in_TSS,
                good_seamanship = good_seamanship
            )

            if self.path_planner.has_path(): 
                # Compute optimal path
                path: PWLPath = self.path_planner.get_dijkstra_path()

                # Parameterize in time to get trajectory 
                traj: PWLTrajectory = self.projector.add_timestamps(path)

                logger.info(f"speed and heading required for COLAV: {traj.get_speed(0):.1f} [m/s], {traj.get_heading(0, degrees=True):.1f} [deg]")

                return traj, {
                    'pf': pf,
                    'projected_obstacles': projected_obstacles,
                    'shore': self.shore,
                    'trajectory': traj
                }  


        # if max_iter was reached, returns None
        logger.warning(f"Max number of iterations reached: no valid trajectory was found. Try decreasing the speed_factor or increasing the max number of iterations")
        return None, {
            'projected_obstacles': projected_obstacles,
            'shore': self.shore
        }
    
if __name__ == "__main__":
    import colav, logging
    colav.configure_logging(level=logging.DEBUG)
    planner = colav.TimeSpaceColav(3)    