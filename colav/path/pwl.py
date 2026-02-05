"""
Piece-Wise-Linear (PWL) path and trajectory representations.

Provides classes for maritime path planning and collision detection.

Key classes:
- PWLPath: 2D spatial paths
- PWLTrajectory: 3D space-time trajectories with collision analysis
"""

from typing import List, Tuple, Optional
from shapely import LineString, Point
from matplotlib.axes import Axes
import matplotlib.pyplot as plt, numpy as np

class PWLPath:
    """
    Piecewise linear path in 2D space.
    
    Represents a maritime path as a sequence of waypoints connected by straight 
    line segments. Provides geometric operations for spatial path analysis 
    without temporal constraints.
    
    Parameters
    ----------
    xy : list of tuple
        Sequence of waypoints as (x, y) coordinate tuples in meters.
        Must contain at least 2 points to form a valid path.
        
    Attributes
    ----------
    xy : list of tuple
        Original waypoint coordinates
    length : float
        Total path length in meters
        
    Methods
    -------
    interpolate(distance, normalized=False)
        Get position at specified distance along path
    get_closest_point(x, y)
        Find closest point on path to given coordinates
    progression(x, y, normalized=False)
        Calculate distance along path to closest point projection
    plot(ax=None, **kwargs)
        Plot path on matplotlib axes
    scatter(ax=None, **kwargs)
        Plot waypoints as scatter plot
        
    Examples
    --------
    Create a rectangular path:
    
    >>> waypoints = [(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]
    >>> path = PWLPath(waypoints)
    >>> print(f"Total distance: {path.length:.1f} m")
    
    Find position at halfway point:
    
    >>> x, y = path.interpolate(path.length * 0.5)
    >>> print(f"Midpoint: ({x:.1f}, {y:.1f})")
    
    Track vessel progress along path:
    
    >>> vessel_pos = (5, 2.5)  # Current vessel position
    >>> progress = path.progression(*vessel_pos, normalized=True)
    >>> print(f"Vessel is {progress:.1%} along the path")
    
    Notes
    -----
    All geometric operations use Shapely LineString for robustness and efficiency.
    Coordinates are assumed to be in a projected coordinate system (e.g., UTM)
    with units in meters.
    """
    def __init__(self, xy: List[Tuple[float, float]]):
        self._linestring = LineString(xy)

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Find closest point on path to given coordinates.
        
        Parameters
        ----------
        x, y : float
            Target coordinates in meters.
            
        Returns
        -------
        tuple of float
            Closest point (x, y) on path.
        """
        distance = self._linestring.project(Point(x, y))
        closest = self._linestring.interpolate(distance)
        return closest.x, closest.y

    def interpolate(self, distance: float, normalized: bool = False) -> Tuple:
        """
        Get position at specified distance along path.
        
        Parameters
        ----------
        distance : float
            Distance along path. Units depend on normalized parameter.
        normalized : bool, default False
            If True, distance is fraction [0,1]. If False, distance in meters.
            
        Returns
        -------
        tuple of float
            Position (x, y) at specified distance.
        """
        point = self._linestring.interpolate(distance, normalized=normalized)
        return point.x, point.y
    
    def progression(self, x: float, y: float, normalized: bool = False) -> float:
        """
        Calculate distance along path to closest point projection.
        
        Projects given coordinates onto path and returns progression distance.
        Useful for tracking vessel progress along planned routes.
        
        Parameters
        ----------
        x, y : float
            Coordinates to project onto path (meters).
        normalized : bool, default False
            If True, return fraction [0,1]. If False, return distance in meters.
            
        Returns
        -------
        float
            Distance along path to closest point.
        """
        prog = self._linestring.project(Point(x, y))
        return prog / self._linestring.length if normalized else prog
    
    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """
        Plot path as connected line segments.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments passed to matplotlib plot function.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    def scatter(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """
        Plot path waypoints as scatter plot.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments passed to matplotlib scatter function.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    @property
    def xy(self) -> List[Tuple[float, float]]:
        return [(point[0], point[1]) for point in self._linestring.coords]
    
    @property
    def length(self) -> float:
        return self._linestring.length


class PWLTrajectory(PWLPath):
    """
    Piecewise linear trajectory in 3D timespace.
    
    Extends PWLPath to include temporal information, enabling analysis of vessel
    motion over time. Critical for collision avoidance algorithms that must
    consider timing constraints and relative motion between vessels.
    
    Parameters
    ----------
    xyt : list of tuple
        Sequence of waypoints as (x, y, time) tuples. Spatial coordinates
        in meters, time in seconds. Time values must be monotonically increasing.
        
    Attributes
    ----------
    xyt : list of tuple
        Original waypoint coordinates with timestamps
    waypoints : list of tuple  
        Spatial coordinates only (x, y) without time
    xy : list of tuple
        Inherited from PWLPath - spatial waypoints
    length : float
        Inherited from PWLPath - total spatial length in meters
        
    Methods
    -------
    __call__(t)
        Get 2D position at specified time (makes trajectory callable)
    interpolate(distance, normalized=False)
        Get (x, y, time) at specified distance along trajectory
    get_heading(time, degrees=False)
        Calculate heading/bearing at given time
    get_speed(time)
        Calculate instantaneous speed at given time
    get_edge_speed(p1, p2)
        Calculate speed between two trajectory points
    compute_cpa(other, dt=1.0)
        Compute DCPA/TCPA with another trajectory
        
    Inherited Methods
    -----------------
    All spatial methods from PWLPath: get_closest_point, progression,
    plot, scatter
        
    Examples
    --------
    Create trajectory for vessel moving east then north:
    
    >>> waypoints = [(0, 0, 0), (100, 0, 50), (100, 100, 120)]
    >>> traj = PWLTrajectory(waypoints)
    
    Get vessel position at specific time:
    
    >>> x, y = traj(75)  # Position at t=75 seconds
    >>> print(f"At t=75s: ({x:.1f}, {y:.1f})")
    
    Analyze vessel motion characteristics:
    
    >>> # Heading at t=60 seconds
    >>> heading = traj.get_heading(60, degrees=True)
    >>> print(f"Heading: {heading:.1f}°")
    >>> 
    >>> # Speed at t=30 seconds
    >>> speed = traj.get_speed(30)
    >>> print(f"Speed: {speed:.2f} m/s")
    
    Use for collision detection:
    
    >>> # Check if two vessels will be close at same time
    >>> t_conflict = 90  # seconds
    >>> pos1 = traj1(t_conflict)
    >>> pos2 = traj2(t_conflict) 
    >>> distance = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
    >>> if distance < safe_distance:
    >>>     print("Collision risk detected!")
    
    Notes
    -----
    - Trajectory assumes linear motion between waypoints
    - Time interpolation uses linear approximation
    - Speed calculations use finite differences for numerical stability
    - Inherits all spatial operations from PWLPath base class
    - Essential component for timespace collision avoidance algorithms
    
    See Also
    --------
    PWLPath : Base class for spatial path operations
    TimeSpaceProjector : Projects trajectories for collision analysis
    """
    def __init__(self, xyt: List[Tuple[float, float, float]]):
        # Extract spatial coordinates for base class 
        super().__init__([(point[0], point[1]) for point in xyt])
        # Override with 3D linestring for time operations
        self._linestring = LineString(xyt)

    def __call__(self, t: float) -> Tuple[float, float]:
        """
        Get 2D position at specified time (callable interface).
        
        Enables trajectory(time) syntax for intuitive position queries.
        Essential for collision detection algorithms.
        
        Parameters
        ----------
        t : float
            Time in seconds.
            
        Returns
        -------
        tuple of float
            Position (x, y) in meters at time t.
            
        Raises
        ------
        ValueError
            If time is outside trajectory bounds.
        """
        coords = list(self._linestring.coords)
        
        # Find segment and interpolate
        for i in range(len(coords) - 1):
            t1, t2 = coords[i][2], coords[i+1][2]
            if t1 <= t <= t2:
                # Linear interpolation
                ratio = (t - t1) / (t2 - t1) if t2 != t1 else 0
                x = coords[i][0] + ratio * (coords[i+1][0] - coords[i][0])
                y = coords[i][1] + ratio * (coords[i+1][1] - coords[i][1])
                return (x, y)
        
        raise ValueError(f"Time {t} outside trajectory bounds")

    def interpolate(self, distance: float, normalized=False) -> Tuple[float, float, float]:
        """
        Get position and time at specified distance along trajectory.
        
        Overrides PWLPath.interpolate to include time dimension.
        
        Parameters
        ----------
        distance : float
            Distance along trajectory path.
        normalized : bool, default False
            If True, distance is fraction [0,1]. If False, distance in meters.
            
        Returns
        -------
        tuple of float
            Position and time (x, y, t) at specified distance.
        """
        point = self._linestring.interpolate(distance, normalized=normalized)
        return point.x, point.y, point.z
    
    def get_heading(self, time: float, degrees: bool = False) -> float:
        """
        Calculate heading (bearing) at specified time along trajectory.
        
        Uses finite differences for numerical stability at waypoints.
        Essential for vessel control and collision avoidance.
        
        Parameters
        ----------
        time : float
            Time in seconds to calculate heading.
        degrees : bool, default False
            If True, return in degrees. If False, return in radians.
            
        Returns
        -------
        float
            Heading angle. North=0°, East=90° (nautical convention).
        """
        coords = list(self._linestring.coords)
        t_start, t_end = coords[0][2], coords[-1][2]
        
        assert t_start <= time <= t_end, f"time must be within [{t_start:.1f}, {t_end:.1f}]. Got time={time:.2f}"
        
        # Use small time offset for finite difference
        dt = 0.1  # 0.1 second offset
        
        if time == t_start:
            t_prev = time
            t_next = min(time + dt, t_end)
        elif time == t_end:
            t_prev = max(time - dt, t_start)
            t_next = time
        else:
            t_prev = max(time - dt/2, t_start)
            t_next = min(time + dt/2, t_end)

        p1 = np.array(self(t_prev))
        p2 = np.array(self(t_next))

        heading_rad = np.atan2(p2[0]-p1[0], p2[1]-p1[1])
        return np.rad2deg(heading_rad) if degrees else heading_rad
    
    def get_speed(self, time: float) -> float:
        """
        Calculate instantaneous speed at specified time along trajectory.
        
        Uses finite differences between nearby trajectory points.
        Critical for speed constraint validation.
        
        Parameters
        ----------
        time : float
            Time in seconds to calculate speed.
            
        Returns
        -------
        float
            Speed in meters per second.
        """
        coords = list(self._linestring.coords)
        t_start, t_end = coords[0][2], coords[-1][2]
        
        assert t_start <= time <= t_end, f"time must be within [{t_start:.1f}, {t_end:.1f}]. Got time={time:.2f}"
        
        # Use small time offset for finite difference
        dt = 0.1  # 0.1 second offset
        
        if time == t_start:
            t_prev = time
            t_next = min(time + dt, t_end)
        elif time == t_end:
            t_prev = max(time - dt, t_start)
            t_next = time
        else:
            t_prev = max(time - dt/2, t_start)
            t_next = min(time + dt/2, t_end)

        pos1 = self(t_prev)
        pos2 = self(t_next)
        
        # Create 3D points with time for get_edge_speed
        p1 = (pos1[0], pos1[1], t_prev)
        p2 = (pos2[0], pos2[1], t_next)
        
        return self.get_edge_speed(p1, p2)
    
    def get_edge_speed(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """
        Calculate speed between two trajectory points.
        
        Helper method for speed calculations using spatial distance and time difference.
        
        Parameters
        ----------
        p1, p2 : tuple of float
            Trajectory points as (x, y, time) tuples.
            
        Returns
        -------
        float
            Speed in meters per second between points.
        """
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 / (p2[2] - p1[2])

    def compute_cpa(self, other: "PWLTrajectory", dt: float = 1.0) -> Tuple[float, float]:
        """
        Compute Distance and Time to Closest Point of Approach (DCPA/TCPA).
        
        Calculates the minimum distance between two vessel trajectories and 
        the time when this occurs.
        
        Parameters
        ----------
        other : PWLTrajectory
            Other vessel's trajectory to compare against.
        dt : float, default 1.0
            Time sampling interval in seconds for numerical computation.
            
        Returns
        -------
        tuple of float
            (DCPA, TCPA) where:
            - DCPA: Distance of Closest Point of Approach in meters
            - TCPA: Time to Closest Point of Approach in seconds
            
        Raises
        ------
        ValueError
            If trajectories have no overlapping time period.
            
        Examples
        --------
        >>> # Two crossing vessel trajectories
        >>> traj1 = PWLTrajectory([(0, 0, 0), (100, 0, 100)])    # East
        >>> traj2 = PWLTrajectory([(50, -50, 0), (50, 50, 100)]) # North  
        >>> dcpa, tcpa = traj1.compute_cpa(traj2)
        >>> print(f"Closest approach: {dcpa:.1f}m at t={tcpa:.1f}s")
        
        Notes
        -----
        Uses numerical sampling over overlapping time period. For higher
        accuracy, decrease dt parameter at cost of computation time.
        """
        # Get time bounds for both trajectories
        coords1 = list(self._linestring.coords)
        coords2 = list(other._linestring.coords)
        
        t1_start, t1_end = coords1[0][2], coords1[-1][2]
        t2_start, t2_end = coords2[0][2], coords2[-1][2]
        
        # Find overlapping time period
        t_start = max(t1_start, t2_start)
        t_end = min(t1_end, t2_end)
        
        if t_start >= t_end:
            raise ValueError("Trajectories have no overlapping time period")
        
        # Sample trajectories and find minimum distance
        min_distance = float('inf')
        cpa_time = t_start
        
        current_time = t_start
        while current_time <= t_end:
            try:
                pos1 = self(current_time)
                pos2 = other(current_time)
                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    cpa_time = current_time
                    
            except ValueError:
                # Time outside one of the trajectories
                pass
                
            current_time += dt
        
        return min_distance, cpa_time

    @property
    def xyt(self) -> List[Tuple[float, float, float]]:
        """Get trajectory waypoints with time"""
        return [(point[0], point[1], point[2]) for point in self._linestring.coords]
    
    @property
    def waypoints(self) -> List[Tuple[float, float]]:
        """Get spatial waypoints without time"""
        return [(point[0], point[1]) for point in self._linestring.coords]

if __name__ == "__main__":
    from colav.timespace.projector import TimeSpaceProjector
    wpts = [
        (0, 0, 0),
        (2, 1, 1),
        (2.5, 0.5, 1.5),
        (3, 0, 2.5)
    ]

    path = PWLPath([(wpt[0], wpt[1]) for wpt in wpts])
    traj = PWLTrajectory(wpts)
    proj = TimeSpaceProjector(3)
    print(proj.get((0, 0), (2, 0), []))
    print(proj.add_timestamps(path).xyt)
    print(path.interpolate(0.5, normalized=True), traj._linestring.interpolate(0.5, normalized=True))
    print(path.xy, traj.xyt)

    # Two crossing vessels
    traj1 = PWLTrajectory([(0, 0, 0), (100, 0, 100)])    # Moving east
    traj2 = PWLTrajectory([(50, -50, 0), (50, 50, 150)]) # Moving north

    dcpa, tcpa = traj1.compute_cpa(traj2)
    print(f"Vessels will be {dcpa:.1f}m apart at t={tcpa:.1f}s")
