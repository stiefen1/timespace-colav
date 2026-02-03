"""
Timespace projection for moving obstacle collision avoidance.

Provides TimeSpaceProjector class for converting moving obstacles into
static polygon representations, enabling the use of standard path planning
algorithms for collision avoidance.

Key Components
--------------
TimeSpaceProjector : Projects moving obstacles into static polygons

Notes
-----
Timespace projection transforms the dynamic collision avoidance problem
into a static geometric path planning problem by intersecting the obstacle
future positions with a timespace plane.
"""

from colav.timespace.plane import Plane
from colav.obstacles.moving import MovingObstacle
from colav.path.pwl import PWLTrajectory, PWLPath
from typing import List, Tuple, Optional
from shapely import Polygon
import numpy as np, logging
logger = logging.getLogger(__name__)
class TimeSpaceProjector:
    """
    Timespace projector for moving obstacle collision avoidance.
    
    Converts moving obstacles into static polygon representations. This transformation
    enables standard geometric path planning algorithms to handle dynamic environments.
    
    Parameters
    ----------
    v_des : float
        Desired navigation speed in m/s to design timespace plane.
        
    Attributes
    ----------
    v_des : float
        Current desired speed setting.
    plane : Plane or None
        Active timespace plane (created during projection).
        
    Methods
    -------
    get(p, p_des, obstacles)
        Convert moving obstacles into static polygons
    add_timestamps(path)
        Add timing information to geometric path based on timespace plane
        
    Examples
    --------
    Basic obstacle projection:
    
    >>> projector = TimeSpaceProjector(v_des=10.0)
    >>> static_obstacles = projector.get(
    ...     (0, 0), (100, 100), [moving_ship1, moving_ship2]
    ... )
    
    Add timing to path:
    
    >>> path = PWLPath([(0,0), (50,50), (100,100)])
    >>> trajectory = projector.add_timestamps(path)
    >>> print(trajectory.get_speed(0))  # Speed at first segment
    
    Adjust speed during planning:
    
    >>> projector.v_des = 8.0  # Reduce speed
    >>> updated_obstacles = projector.get(start, target, ships)
    
    Notes
    -----
    The projection creates a timespace plane where:
    - Distance from start to target and desired speed determines plane geometry
    - Moving obstacles are intersected with plane to get static polygons
    - Only obstacles with future intersections are included
    
    Timespace projection is fundamental to the collision avoidance approach,
    transforming dynamic problems into solvable geometric optimization.
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
        Project moving obstacles into static polygons.
        
        Creates timespace plane from start to target points and projects
        moving obstacle vertices onto the plane to generate static polygon
        representations for collision-free path planning.
        
        Parameters
        ----------
        p : tuple of float
            Start position (x, y) in meters.
        p_des : tuple of float
            Target position (x, y) in meters.
        obstacles : list of MovingShip
            Moving obstacles to project.
            
        Returns
        -------
        list of shapely.Polygon
            Static polygon obstacles in timespace projection.
            Only includes obstacles with future intersections.
            
        Notes
        -----
        - Creates internal timespace plane based on desired speed
        - Projects each obstacle vertex using its velocity
        - Filters out obstacles entirely in the past
        
        Examples
        --------
        >>> ship = MovingShip((50, 0), 90, (5, 0), 8, 3, mmsi=123)
        >>> static_obs = projector.get((0, 0), (100, 100), [ship])
        >>> print(f"Generated {len(static_obs)} static obstacles")
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
        """
        Convert geometric path to trajectory.
        
        Uses the internal timespace plane to add timing information
        to a geometric path, creating a trajectory with speed profile.
        
        Parameters
        ----------
        path : PWLPath
            Geometric piecewise-linear (PWL) path to add timing to.
            
        Returns
        -------
        PWLTrajectory
            PWL Trajectory with (x, y, t) waypoints.
            
        Raises
        ------
        ValueError
            If no timespace plane exists (call get() first).
            
        Examples
        --------
        >>> path = PWLPath([(0,0), (50,50), (100,100)])
        >>> trajectory = projector.add_timestamps(path)
        >>> print(trajectory.get_heading(0, degrees=True))
        """
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