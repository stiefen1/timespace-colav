
"""
Timespace collision avoidance planning for maritime navigation.

Provides high-level interface for maritime collision avoidance using timespace
projection and visibility graph path planning. Integrates COLREGS compliance,
speed constraints, and iterative optimization for safe trajectory generation.

Key Components
--------------
TimeSpaceColav : class
    Main collision avoidance planner with configurable constraints
    
Notes
-----
Designed for maritime autonomous navigation with moving obstacles.
Supports COLREGS compliance and good seamanship practices.
Uses iterative speed reduction to find feasible collision-free paths.
"""

from typing import List, Tuple, Optional, Dict
from colav.obstacles import MovingObstacle, MovingShip
from colav.timespace.projector import TimeSpaceProjector
from colav.path.planning import PathPlanner
from colav.path.pwl import PWLPath, PWLTrajectory
from colav.path.planning import VGPathPlanner
from colav.timespace.constraints import SpeedConstraint, CourseRateConstraint, COLREGS
import shapely, logging
from colav.utils.math import DEG2RAD, RAD2DEG, ssa
from math import atan2, cos, sin
logger = logging.getLogger(__name__)

class TimeSpaceColav:
    """
    Timespace collision avoidance planner for maritime navigation.
    
    Combines timespace projection with visibility graph path planning to generate
    collision-free trajectories that comply with maritime regulations. Uses iterative
    optimization to find feasible paths under speed and course rate constraints.
    
    Parameters
    ----------
    desired_speed : float
        Preferred navigation speed in m/s. Must be positive.
    distance_threshold : float, default 3000
        Distance in meters at which collision avoidance is activated.
        Obstacles beyond this range are ignored.
    shore : list of Polygon, optional
        Static shore obstacles to avoid. Each polygon represents
        a land mass or restricted area.
    max_speed : float, default inf
        Maximum allowable speed in m/s for generated trajectories.
    max_course_rate : float, default inf
        Maximum allowable course change rate in deg/s (if degrees=True)
        or rad/s (if degrees=False).
    colregs : bool, default False
        Enable COLREGS (International Regulations for Preventing
        Collisions at Sea) compliance in path planning.
    good_seamanship : bool, default False
        Enable good seamanship practices beyond basic COLREGS.
    path_planner : PathPlanner, optional
        Custom path planner instance. Uses default VGPathPlanner if None.
    max_iter : int, default 10
        Maximum optimization iterations before giving up.
    speed_factor : float, default 0.95
        Speed reduction factor for iterative optimization.
        Must be in (0, 1]. At iteration k, speed *= speed_factor^k.
    degrees : bool, default True
        Whether angular measurements are in degrees (True) or radians (False).
    abort_colregs_after_iter : int, optional
        Iteration after which COLREGS compliance is disabled to find
        any feasible solution. Defaults to max_iter // 2.
        
    Attributes
    ----------
    desired_speed : float
        Target navigation speed in m/s.
    distance_threshold : float
        Activation distance for collision avoidance.
    shore : list of Polygon
        Static shore obstacles.
    max_speed, max_course_rate : float
        Speed and course rate constraints.
    colregs : bool
        COLREGS compliance flag.
    projector : TimeSpaceProjector
        Timespace projection engine.
    path_planner : PathPlanner
        Underlying geometric path planner.
    edge_filters, node_filters : list
        Constraint filters for graph construction.
        
    Methods
    -------
    get(p0, pf, obstacles, **kwargs)
        Generate collision-free trajectory between waypoints
        
    Examples
    --------
    Basic collision avoidance setup:
    
    >>> from colav.obstacles import MovingShip
    >>> planner = TimeSpaceColav(
    ...     desired_speed=10.0,  # 10 m/s
    ...     max_speed=15.0,
    ...     max_course_rate=5.0,  # 5 deg/s
    ...     colregs=True
    ... )
    
    Plan trajectory with moving obstacles:
    
    >>> ship = MovingShip((100, 0), 90, (5, 0), 8, 3, mmsi=123)
    >>> trajectory, info = planner.get(
    ...     p0=(0, 0),
    ...     pf=(200, 100), 
    ...     obstacles=[ship],
    ...     heading=45.0  # Current vessel heading
    ... )
    
    With shore constraints:
    
    >>> from shapely import Polygon
    >>> shore_polygon = Polygon([(50,50), (150,50), (150,150), (50,150)])
    >>> planner = TimeSpaceColav(
    ...     desired_speed=8.0,
    ...     shore=[shore_polygon],
    ...     colregs=True,
    ...     good_seamanship=True
    ... )
    
    Notes
    -----
    - Uses iterative speed reduction to find feasible solutions
    - COLREGS compliance can be disabled mid-optimization if needed
    - Returns None if no collision-free path found within max_iter
    - Trajectory timing based on projected minimum speeds
    - Supports both Traffic Separation Scheme (TSS) compliance
    - Handles complex multi-ship encounter scenarios
    
    Performance considerations:
    - Computation time scales with number of obstacles and iterations
    - Use appropriate distance_threshold to limit obstacle count
    - Consider obstacle geometry simplification for complex shapes
    
    See Also
    --------
    TimeSpaceProjector : Timespace obstacle projection
    VGPathPlanner : Visibility graph path planning
    MovingShip : Moving obstacle representation
    COLREGS : Maritime collision regulations filter
    """
    
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
            CourseRateConstraint(course_rate=DEG2RAD(max_course_rate) if degrees else max_course_rate)
        ]
        self.node_filters = [COLREGS(good_seamanship=good_seamanship)] if colregs else []

        logger.debug(f"Successfully initialized TimeSpaceColav")

    def get(
            self, 
            p0: Tuple[float, float],
            pf: Tuple[float, float],
            obstacles: List[MovingShip], 
            *args, 
            heading: Optional[float] = None, 
            margin: float = 0.0, 
            degrees: bool = True,
            ts_in_TSS: bool = False,
            os_in_TSS: bool = False,
            good_seamanship: bool = False,
            **kwargs
        ) -> Tuple[Optional[PWLTrajectory], Dict]:
        """
        Generate collision-free trajectory between waypoints.
        
        Computes optimal trajectory from start to target point
        while avoiding moving obstacles and respecting maritime constraints.
        Uses iterative optimization with speed reduction if initial plans fail.
        
        Parameters
        ----------
        p0 : tuple of float
            Start waypoint coordinates (x, y) in meters.
        pf : tuple of float
            Target waypoint coordinates (x, y) in meters.
        obstacles : list of MovingShip
            Moving obstacles to avoid. Only obstacles within distance_threshold
            of p0 will be considered for collision avoidance.
        heading : float, optional
            Current vessel heading in degrees (if degrees=True) or radians.
            Used for course rate constraint enforcement if max_course_rate
            was specified during initialization.
        margin : float, default 0.0
            Safety margin in meters applied to obstacle boundaries through
            Shapely buffer operation. Positive values increase obstacle size.
        degrees : bool, default True
            Whether angular measurements (heading) are in degrees (True)
            or radians (False).
        ts_in_TSS : bool, default False
            Whether target ship (own ship) is operating in Traffic Separation
            Scheme waters. Affects COLREGS compliance behavior.
        os_in_TSS : bool, default False
            Whether other ships are operating in Traffic Separation Scheme
            waters. Affects encounter classification and rule application.
        good_seamanship : bool, default False
            Enable good seamanship practices beyond basic COLREGS compliance.
            Overrides class-level setting for this specific planning request.
        **kwargs
            Additional keyword arguments passed to Shapely buffer() method
            for obstacle margin application (e.g., cap_style, join_style).
            
        Returns
        -------
        trajectory : PWLTrajectory or None
            Piecewise-linear (PWL) trajectory from p0 to pf.
            Returns None if no collision-free path found within max_iter.
        info : dict
            Planning information and intermediate results:
            
            - 'pf': Target waypoint coordinates (only if successful)
            - 'projected_obstacles': Timespace footprint of the moving obstacles
            - 'shore': Shore obstacle polygons
            - 'trajectory': Generated trajectory (only if successful)
            
        Raises
        ------
        AssertionError
            If heading is None when max_course_rate constraint is active.
            
        Warnings
        --------
        Issues warnings for:
        - Heading significantly different from target direction (>90°)
        - COLREGS compliance disabled during optimization
        - Maximum iterations reached without finding solution
        
        Notes
        -----
        Algorithm steps:
        1. Filter obstacles by distance_threshold from start point
        2. Apply safety margin through buffer operation
        3. Project moving obstacles into timespace plane
        4. Construct visibility graph with constraints
        5. Find shortest collision-free path
        6. Parameterize path with timestamps
        7. If no solution: reduce speed and retry (up to max_iter)
        
        Speed reduction follows: v_k = v_desired / (1 + 5*k/max_iter)
        where k is the current iteration number.
        
        COLREGS compliance may be disabled after abort_colregs_after_iter
        to find any feasible solution when strict compliance prevents
        collision avoidance.
        
        Examples
        --------
        Basic trajectory planning:
        
        >>> ship = MovingShip((100, 50), 180, (0, -5), 8, 3, mmsi=456)
        >>> trajectory, info = planner.get(
        ...     p0=(0, 0),
        ...     pf=(200, 100),
        ...     obstacles=[ship],
        ...     heading=30.0,
        ...     margin=50.0  # 50m safety margin
        ... )
        >>> if trajectory:
        ...     print(f"Speed: {trajectory.get_speed(0):.1f} m/s")
        ...     print(f"Initial heading: {trajectory.get_heading(0):.1f}°")
        
        Multi-ship encounter with TSS:
        
        >>> ships = [ship1, ship2, ship3]
        >>> trajectory, info = planner.get(
        ...     p0=(0, 0), pf=(1000, 500),
        ...     obstacles=ships,
        ...     ts_in_TSS=True,
        ...     good_seamanship=True,
        ...     margin=100.0
        ... )
        
        Handle planning failure:
        
        >>> trajectory, info = planner.get(p0, pf, obstacles)
        >>> if trajectory is None:
        ...     print("No collision-free path found")
        ...     # Consider: reduce speed, increase iterations,
        ...     #          disable COLREGS, or change waypoints
        
        See Also
        --------
        PWLTrajectory : Returned trajectory type with timing information
        TimeSpaceProjector : Projects moving obstacles into timespace
        VGPathPlanner : Underlying visibility graph path planner
        MovingShip : Moving obstacle representation and methods
        """
        desired_heading = atan2(pf[0]-p0[0], pf[1]-p0[1])

        if heading is not None:
            heading = DEG2RAD(heading) if degrees else heading

            assert heading is not None, f"heading is None."
            angle_error = ssa(heading - desired_heading)

            # Raise warning if heading is very different from the actual direction leading to pf
            if abs(angle_error) >= DEG2RAD(90):
                logger.warning(f"Heading angle is {RAD2DEG(heading):.0f} degrees, but target point is oriented towards {RAD2DEG(desired_heading):.0f} degrees")

        buffered_obstacles: List[MovingShip] = []
        for obs in obstacles:
            if obs.distance(*p0) <= self.distance_threshold:
                buffered_obstacles.append(obs.buffer(margin, **kwargs) if margin > 0 else obs)

        discount_power = 0
        colregs_active = self.colregs
        projected_obstacles = []
        for k in range(self.max_iter):
            # Decrease desired speed at each iteration to find a plane that admits at least one feasible path
            # Alternative method: self.projector.v_des = (self.speed_factor**discount_power) * self.desired_speed
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
                path: PWLPath = self.path_planner.get()

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