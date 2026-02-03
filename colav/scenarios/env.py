"""
Simulation environment for maritime collision avoidance scenarios.

Provides realistic simulation environment for testing timespace collision
avoidance algorithms with multiple vessels, static obstacles, and maritime
regulations in dynamic scenarios.

Key Components
--------------
COLAVEnv : Complete simulation environment for collision avoidance testing

Notes
-----
Designed for demonstration and validation of collision avoidance frameworks
with simplified maritime physics, multiple ship interactions, and regulatory
compliance testing.
"""

from colav.path.pwl import PWLPath
from colav.obstacles.moving import MovingShip
from colav.planner import TimeSpaceColav
from colav.path.planning import PathPlanner
from colav.utils.math import DEG2RAD
from shapely import Polygon
from typing import List, Optional, Tuple, Dict, Any

MAX_COURSE_RATE_DEGS = 1      # Maximum course rate in deg/s
MAX_ACC_MS2 = 0.05           # Maximum acceleration in m/s^2


class COLAVEnv:
    """
    Maritime collision avoidance simulation environment.
    
    Complete simulation environment that integrates vessel dynamics, collision
    avoidance planning, static/moving obstacles, and maritime regulations for
    testing and demonstrating autonomous navigation algorithms.
    
    Parameters
    ----------
    own_ship : MovingShip
        The vessel under control that will follow the planned path.
    path : PWLPath
        Global reference path for the own ship to follow.
    desired_speed : float
        Target navigation speed in m/s for collision avoidance planning.
    obstacles : list of MovingShip, optional
        Other vessels in the environment that must be avoided.
    distance_threshold : float, default 3000
        Distance in meters at which collision avoidance activates.
    shore : list of Polygon, optional
        Static shore obstacles and restricted areas.
    max_speed : float, default inf
        Maximum allowable speed for the own ship.
    max_course_rate : float, default inf
        Maximum course change rate in deg/s (or rad/s if degrees=False).
    colregs : bool, default False
        Enable COLREGS compliance in collision avoidance.
    good_seamanship : bool, default False
        Enable good seamanship practices beyond basic COLREGS.
    path_planner : PathPlanner, optional
        Custom path planner. Uses default (VGPathPlanner) if None.
    max_iter : int, default 10
        Maximum iterations for collision avoidance.
    speed_factor : float, default 0.95
        Speed reduction factor for iterative planning.
    degrees : bool, default True
        Whether angular units are in degrees.
    lookahead_distance : float, default 300
        Distance ahead along the path to select target point
    buffer_moving : float, default 0
        Safety buffer added to moving obstacles in meters.
    simplify_moving : float, default 0
        Geometry simplification tolerance for moving obstacles.
    buffer_static : float, default 0
        Safety buffer added to static obstacles in meters.
    simplify_static : float, default 0
        Geometry simplification tolerance for static obstacles.
    abort_colregs_after_iter : int, optional
        Iteration after which to disable COLREGS for any solution.
        
    Attributes
    ----------
    own_ship : MovingShip
        Current state of the controlled vessel.
    obstacles : list of MovingShip
        Current states of all other vessels.
    path : PWLPath
        Reference path being followed.
    planner : TimeSpaceColav
        Collision avoidance planner instance.
        
    Methods
    -------
    step(dt)
        Advance simulation by one time step
        
    Examples
    --------
    Basic encounter scenario:
    
    >>> from colav.obstacles import MovingShip
    >>> from colav.path.pwl import PWLPath
    >>> from shapely import Polygon
    >>> 
    >>> own_ship = MovingShip.from_body((0, 0), 45, 5, 0, 20, 5)
    >>> obstacle = MovingShip.from_body((100, 50), 180, 4, 0, 15, 4)
    >>> path = PWLPath([(0, 0), (200, 200)])
    >>> 
    >>> env = COLAVEnv(
    ...     own_ship=own_ship,
    ...     path=path,
    ...     desired_speed=5.0,
    ...     obstacles=[obstacle],
    ...     colregs=True,
    ...     max_speed=8.0,
    ...     buffer_moving=50.0
    ... )
    
    Multi-ship scenario with shore:
    
    >>> shore_polygon = Polygon([(50,50), (150,50), (150,150), (50,150)])
    >>> env = COLAVEnv(
    ...     own_ship=own_ship,
    ...     path=path,
    ...     desired_speed=5.0,
    ...     obstacles=[ship1, ship2, ship3],
    ...     shore=[shore_polygon],
    ...     colregs=True,
    ...     good_seamanship=True,
    ...     lookahead_distance=500
    ... )
    
    Notes
    -----
    Environment automatically:
    - Updates vessel positions and dynamics each step
    - Plans collision-free paths
    - Applies realistic acceleration and course rate limits
    - Resamples obstacle velocities for dynamic behavior
    - Tracks path progression for completion detection
    
    Designed for testing various encounter scenarios:
    - Head-on encounters
    - Crossing situations
    - Overtaking maneuvers
    - Multi-ship encounters
    - Navigation in restricted waters
    
    See Also
    --------
    ScenarioRunner : For running and visualizing complete scenarios
    TimeSpaceColav : Underlying collision avoidance algorithm
    MovingShip : Vessel dynamics and representation
    """
    
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
        """
        Advance simulation by one time step.
        
        Performs collision avoidance planning, updates vessel dynamics,
        and advances all entities in the simulation environment.
        
        Parameters
        ----------
        dt : float
            Time step duration in seconds.
            
        Returns
        -------
        observation : dict
            Current simulation state containing:
            - 'own_ship': Current own ship state
            - 'obstacles': List of current obstacle states
            - 'trajectory': Planned collision-free trajectory (or None)
        done : bool
            True if own ship has reached the end of the reference path.
        info : dict
            Additional planning information from collision avoidance.
            
        Notes
        -----
        Step sequence:
        1. Determine current position along reference path
        2. Plan collision-free trajectory to lookahead point
        3. Extract desired speed and heading from trajectory
        4. Apply course rate and acceleration limits
        5. Update own ship state with new commands
        6. Predict forward all vessel positions
        7. Resample target ships velocities for dynamic behavior
        
        The method implements realistic vessel dynamics including:
        - Course rate limiting (±2 deg/s)
        - Acceleration limiting (±0.02 m/s²)
        - Integration of vessel state over time step
        
        Examples
        --------
        >>> obs, done, info = env.step(dt=1.0)
        >>> if obs['trajectory'] is not None:
        ...     print(f"Speed: {obs['trajectory'].get_speed(0):.1f} m/s")
        >>> if done:
        ...     print("Reached destination!")
        """
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
        max_course_rate = MAX_COURSE_RATE_DEGS if self.own_ship.degrees else DEG2RAD(MAX_COURSE_RATE_DEGS)
        course_rate = max(min(desired_course_rate, max_course_rate), -max_course_rate)
        
        # Limit acceleration
        desired_acceleration = (speed - self.own_ship.u) / dt
        acc = max(min(desired_acceleration, MAX_ACC_MS2), -MAX_ACC_MS2)

        # Integrate
        self.own_ship = MovingShip.from_body(self.own_ship.position, self.own_ship.psi + course_rate * dt, self.own_ship.u + acc * dt, 0, self.own_ship.loa, self.own_ship.beam, degrees=self.own_ship.degrees, mmsi=self.own_ship.mmsi)

        self.own_ship = self.own_ship.predict(dt)
        for i in range(len(self.obstacles)):
            self.obstacles[i] = self.obstacles[i].resample_velocity().predict(dt)

        return {'own_ship': self.own_ship, 'obstacles': self.obstacles, 'trajectory': traj}, self.path.progression(*self.own_ship.position) >= self.path.length, info