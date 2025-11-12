from colav.timespace.plane import Plane
from colav.obstacles.obstacle import MovingObstacle
from typing import List, Tuple

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
    _plane: Plane | None = None

    def __init__(
            self,
            v_des: float
    ):
        assert v_des > 0, f"Desired speed must be > 0. Got {v_des:.1f}"
        self._v_des = v_des

    def get(self, p: Tuple[float, float], p_des: Tuple[float, float], obstacles: List[MovingObstacle]) -> List[ Tuple[float, float] ]:
        """
        Convert a list of moving obstacles into a list of static obstacles as list of vertices.
        The projection is done using a timespace plane designed to reach p_des from p with a desired velocity. 
        """
        
        # Create timespace plane
        dp = ((p_des[1] - p[1])**2 + (p_des[0] - p[0])**2)**0.5
        dt = dp / self._v_des
        self._plane = Plane(p, p_des, 0, dt)

        # Project moving obstacles
        projected_obstacles = []    
        for obs in obstacles:
            # Compute intersection between moving obstacle and timespace plane
            projected_vertices, times = self._plane.intersection(obs.geometry, obs.velocity)
            
            # Check if any intersection occurs in the future
            valid = False
            for t in times:
                if t >= 0:
                    valid = True

            # If at least one intersectio occurs in the future, obstacle is valid
            if valid:
                projected_obstacles.append(projected_vertices)

        return projected_obstacles
    
    @property
    def plane(self) -> Plane | None:
        return self._plane
    
    @property
    def v_des(self) -> float:
        return self._v_des
    
if __name__ == "__main__":
    from colav.obstacles import MovingObstacle, SHIP
    import matplotlib.pyplot as plt

    obs = MovingObstacle.from_body((0, -20), 45, 2**0.5, 0, SHIP(10, 3), degrees=True)
    os = MovingObstacle((-20, -20), 45, (1, 1), SHIP(10, 3), degrees=True)
    projector = TimeSpaceProjector(2)
    projected_obs = projector.get((-20, -20), (20, 20), [obs])

    ax = obs.plot()
    ax.scatter(*zip(*os.geometry))
    ax.scatter(*zip(*projected_obs[0]))
    plt.show()