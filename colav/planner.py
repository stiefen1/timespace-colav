
"""
Typical use case:

"""

from typing import List, Tuple, Optional
from colav.obstacles import MovingObstacle
from colav.timespace.projector import TimeSpaceProjector
from colav.path.planning import PathPlanner
from colav.path.pwl import PWLPath, PWLTrajectory
from colav.path.graph import VisibilityGraph as VG
import shapely

class TimeSpaceColav:
    def __init__(
            self,
            desired_speed: float,
            max_speed: float = float('inf'),
            max_yaw_rate: float = float('inf'),
            colregs: bool = False,
            good_seamanship: bool = False,
            path_planner: Optional[PathPlanner] = None,
            max_iter: int = 10, # Max iterations for finding a valid solution
            speed_factor: float = 0.95 # At each iteration k, speed = speed_factor**k * desired_speed
    ):
        assert desired_speed > 0, f"Desired speed must be > 0. Got {desired_speed} <= 0 instead."
        assert max_speed > 0, f"Maximum speed must be > 0. Got {max_speed} <= 0 instead."
        assert max_yaw_rate > 0, f"Maximum yaw rate must be > 0. Got {max_yaw_rate} <= 0 instead."
        assert 0 < speed_factor <= 1, f"Speed factor must be in ]0, 1]. Got {speed_factor} instead."

        self.desired_speed = desired_speed
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.colregs = colregs
        self.good_seamanship = good_seamanship
        self.projector = TimeSpaceProjector(self.desired_speed)
        self.path_planner = path_planner or PathPlanner()
        self.max_iter = max_iter
        self.speed_factor = speed_factor

    def get(
            self, 
            p0: Tuple[float, float], 
            pf: Tuple[float, float], 
            obstacles: List[MovingObstacle], 
            *args, 
            heading: Optional[float] = None, 
            margin: float = 0.0, 
            **kwargs
        ) -> PWLTrajectory | None:
        """
        Returns the shortest trajectory between p0 and pf to avoid obstacles as a piecewise-linear path parameterized in time.

        If heading is provided and self.max_yaw_rate was provided at initialization, it can be used as a constraint. 

        margin is the dilatation applied to the obstacles through the shapely.buffer() method.

        Keywords argument can be used to affect the shapely.buffer method's behaviour
        """
        buffered_obstacles = [obs.buffer(margin, **kwargs) for obs in obstacles]

        for k in range(self.max_iter):
            # Decrease desired speed at each iteration to find a plane that admits at least one feasible path
            self.projector.v_des = (self.speed_factor**k) * self.desired_speed

            # Get timespace footprint, i.e. static polygons to be avoided
            projected_obstacles: List[shapely.Polygon] = self.projector.get(
                p0,
                pf,
                buffered_obstacles
            )

            # Create path planner
            path_planner = VG(
                p0,
                pf,
                obstacles={obs.mmsi: proj_obs for obs, proj_obs in zip(buffered_obstacles, projected_obstacles)},
                edge_filters=[
                    # e.g. colregs / yaw rate / speed filters
                ]
            )

            if path_planner.has_path(): 
                # Compute optimal path
                path: PWLPath = path_planner.get_dijkstra_path()

                # Parameterize in time to get trajectory 
                traj: PWLTrajectory = self.projector.add_timestamps(path)

                return traj # TODO: Returns necessary speed and heading to reach first waypoint. 


        # if max_iter was reached, returns None
        UserWarning(f"Max number of iterations reached: no valid trajectory was found. Try decreasing the speed_factor or increasing the max number of iterations")
        return None
    
if __name__ == "__main__":
    from colav.obstacles.moving import MovingShip
    planner = TimeSpaceColav(3)
    print(planner.get((0, 0), (100, 100), [MovingShip((20, 20), 30, (2, 2), 10, 4, degrees=True)])[0].geometry)
    