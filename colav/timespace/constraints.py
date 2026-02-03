"""
Constraint filters for maritime path planning.

Provides constraint implementations for enforcing speed limits, course rate
limitations, and COLREGS compliance in visibility graph path planning.

Key Classes
-----------
SpeedConstraint : Enforces maximum speed limits on path edges
CourseRateConstraint : Limits course change rates for vessel maneuverability
COLREGS : Maritime collision regulations compliance filter

Notes
-----
Constraints are applied during visibility graph construction to ensure
generated paths respect vessel capabilities and maritime regulations.
"""

from colav.path.filters import IEdgeFilter, INodeFilter
from colav.timespace.plane import Plane
from colav.obstacles.moving import MovingShip
from typing import Dict, Tuple, Optional
from math import atan2
from colav.utils.math import ssa
from colav.colregs.encounters import get_recommendation_for_os, Recommendation
from colav.utils.mmsi import is_valid_mmsi
import logging, numpy as np
logger = logging.getLogger(__name__)

class SpeedConstraint(IEdgeFilter):
    """
    Speed constraint filter for path planning edges.
    
    Ensures that path segments do not exceed maximum allowable speed
    based on distance and time differences between waypoints.
    
    Parameters
    ----------
    speed : float
        Maximum allowable speed in m/s.
        
    Examples
    --------
    >>> constraint = SpeedConstraint(speed=15.0)  # 15 m/s max
    >>> valid, info = constraint.is_valid(p1, p2, plane)
    """
    
    def __init__(
        self,
        speed: float
    ):
        super().__init__()
        self.speed = speed

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], plane: Plane, **kwargs) -> Tuple[bool, Dict]:
        t2 = plane.get_time(*p2)
        t1 = plane.get_time(*p1)
        info = {}

        if t2 <= t1:
            return False, info

        edge_speed = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5 / (t2 - t1)
        return edge_speed <= self.speed, info
    
class CourseRateConstraint(IEdgeFilter):
    """
    Course rate constraint filter for path planning edges.
    
    Limits the rate of course change based on vessel maneuverability.
    Only applies to edges connected to the starting waypoint.
    
    Parameters
    ----------
    course_rate : float
        Maximum allowable course change rate in rad/s.
        
    Notes
    -----
    Requires vessel heading to be provided during path planning.
    Constraint is only applied to the first edge from start position.
    
    Examples
    --------
    >>> from colav.utils.math import DEG2RAD
    >>> constraint = CourseRateConstraint(DEG2RAD(5))  # 5 deg/s
    >>> valid, info = constraint.is_valid(p1, p2, plane, heading=heading)
    """
    
    def __init__(
        self,
        course_rate: float # radians
    ):
        super().__init__()
        self.course_rate = course_rate

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], plane: Plane, idx1: int, heading: Optional[float] = None, **kwargs) -> Tuple[bool, Dict]:
        """
        heading in radians
        """
        info = {}
        if idx1 != 0: # We only impose this constraint for edges connected to the starting waypoint (index=0)
            return True, info
        
        if heading is None:
            logger.warning(f"Course rate constraint is enabled but the own ship's heading was not provided (course rate constraint will be ignored).")
            return True, info

        # Compute edge speed
        dt = plane.get_time(*p2) - plane.get_time(*p1)
        assert isinstance(dt, float), f"dt must be a float, got dt={dt}"

        if dt <= 0:
            return False, info
        
        edge_angle = atan2(p2[0] - p1[0], p2[1] - p1[1])
        angle_error = ssa(edge_angle - heading)
        course_rate_required = abs(angle_error) / dt
        is_valid = course_rate_required <= self.course_rate
        return is_valid, info

class COLREGS(INodeFilter):
    """
    COLREGS compliance filter for maritime path planning.
    
    Enforces International Regulations for Preventing Collisions at Sea
    by filtering obstacle vertices based on encounter geometry and
    required collision avoidance maneuvers.
    
    Parameters
    ----------
    good_seamanship : bool, default False
        Enable good seamanship practices beyond basic COLREGS.
        
    Notes
    -----
    Determines appropriate turning direction (port/starboard) based on
    encounter type and removes vertices that would require improper
    collision avoidance maneuvers.
    
    Examples
    --------
    >>> colregs = COLREGS(good_seamanship=True)
    >>> valid, info = colregs.is_valid(
    ...     node, start_pos, obstacles_dict, centroid,
    ...     heading=vessel_heading
    ... )
    """
    
    def __init__(
        self,
        good_seamanship: bool = False
    ):
        super().__init__()
        self.good_seamanship = good_seamanship

    def is_valid(
            self,
            node: Dict,
            p_0: Tuple[float, float],
            moving_obstacles_as_dict: Dict[int, MovingShip],
            obstacle_centroid: Tuple[float, float],
            heading: Optional[float] = None,
            ts_in_TSS: bool = False,
            os_in_TSS: bool = False,
            **kwargs
        ) -> Tuple[bool, Dict]:
        info = {'label': ''}

        if heading is None:
            logger.warning(f"COLREGs is enabled but the own ship's heading was not provided (COLREGs will be ignored).")
            return True, info

        # Check whether obstacle is a vessel or not. If not, returns True (valid node)
        if not(is_valid_mmsi(node['id']) or is_valid_mmsi(-node['id'])):
            info['label'] = 'shore'
            return True, info
        
        info['label'] = 'vessel'
        target_ship = moving_obstacles_as_dict[node['id']]
        n = np.array(node['pos']) # node's position

        # Target ship when we own ship reaches node
        ts_timespace_footprint_centroid = np.array(obstacle_centroid)
        os_xy = np.array(p_0)

        dn = n - np.array(p_0)
        dn_unit = dn / np.linalg.norm(dn)
        dp = ts_timespace_footprint_centroid - os_xy
        dp_unit = dp / np.linalg.norm(dp)
        sth = (dp_unit[1] * dn_unit[0] - dp_unit[0] * dn_unit[1])

        reco, info = get_recommendation_for_os(MovingShip(p_0, heading, (0, 0), 0, 0), target_ship, good_seamanship=self.good_seamanship, ts_in_TSS=ts_in_TSS, os_in_TSS=os_in_TSS)

        if reco == Recommendation.TURN_RIGHT:
            if sth < 0:
                logger.debug(f"node removed for COLREGs compliance")
                return False, info
        elif reco == Recommendation.TURN_LEFT:
            if sth > 0:
                logger.debug(f"node removed for COLREGs compliance")
                return False, info
        
        return True, info