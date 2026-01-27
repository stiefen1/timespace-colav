from colav.timespace.plane import Plane
from colav.obstacles.moving import MovingObstacle
from colav.path.pwl import PWLTrajectory, PWLPath
from typing import List, Tuple, Optional
from shapely import Polygon
import numpy as np, logging
logger = logging.getLogger(__name__)
"""
Typical use case:

projector = TimeSpaceProjector(3)
projected_obstacles = projector.get(p, p_des, obstacles)
"""

class TimeSpaceProjector:
    """
    A class to project a collection of moving obstacles into a list of static obstacles.
    """

    _v_des: float
    _plane: Optional[Plane] = None

    def __init__(
            self,
            v_des: float
    ):
        self.v_des = v_des

    def get(self, p: Tuple[float, float], p_des: Tuple[float, float], obstacles: List[MovingShip]) -> List[ Polygon ]:
        """
        Convert a list of moving obstacles into a list of static obstacles as list of vertices.
        The projection is done using a timespace plane designed to reach p_des from p with a desired velocity. 
        """
        logger.debug(f"Received {len(obstacles)} obstacles.")

        # Create timespace plane
        dp = ((p_des[1] - p[1])**2 + (p_des[0] - p[0])**2)**0.5
        dt = dp / self._v_des
        self._plane = Plane(p, p_des, 0, dt)

        # Project moving obstacles
        projected_obstacles = []    
        for obs in obstacles:
            # Compute intersection between moving obstacle and timespace plane
            projected_vertices, times, valid = self._plane.intersection(obs.robust_geometry or obs.geometry, obs.vertices_velocity, robust=obs.robust_geometry is not None)

            # If at least one intersection occurs in the future, obstacle is valid
            if valid:
                projected_obstacles.append(Polygon(projected_vertices))

        logger.debug(f"{len(projected_obstacles)} obstacles have been projected.")

        return projected_obstacles
    
    def add_timestamps(self, path: PWLPath) -> PWLTrajectory:
        if self._plane is not None:
            xy = np.array(path.xy)
            t = self._plane.get_time(xy[:, 0], xy[:, 1])
            assert isinstance(t, np.ndarray), f"xy[:, 0], xy[:, 1] must be a numpy array, Got xy[:, 0]={xy[:, 0]}, xy[:, 1]={xy[:, 1]}"
            xyt = np.concatenate([xy, t[:, None]], axis=1)
            return PWLTrajectory(xyt.tolist())
        else:
            raise ValueError(f"Plane is None. Try calling .get() first to initialize timespace plane.")
    
    @property
    def plane(self) -> Plane | None:
        return self._plane
    
    @property
    def v_des(self) -> float:
        return self._v_des
    
    @v_des.setter
    def v_des(self, val: float) -> None:
        assert val > 0, f"Desired speed must be > 0. Got {val:.1f}"
        self._v_des = val
    
if __name__ == "__main__":
    from colav.obstacles import MovingObstacle, SHIP, MovingShip
    import matplotlib.pyplot as plt

    obs = MovingShip.from_body((20, -40), 0, 2, 0, 10, 3, degrees=True, du=1, dchi=10)
    os = MovingObstacle((-20, -20), 45, (1, 1), SHIP(10, 3), degrees=True)
    projector = TimeSpaceProjector(2)
    projected_obstacles = projector.get((-20, -20), (20, 20), [obs.buffer(1)])

    ax = obs.plot(c='red')
    os.fill(ax=ax, c='blue')
    for proj_obs in projected_obstacles:
        ax.fill(*proj_obs.exterior.xy, c='red', alpha=0.5)
    ax.set_xlim((-80, 80))
    ax.set_ylim((-80, 80))
    plt.show()