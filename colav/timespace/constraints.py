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
    
class YawRateConstraint(IEdgeFilter):
    def __init__(
        self,
        yaw_rate: float # radians
    ):
        super().__init__()
        self.yaw_rate = yaw_rate

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], plane: Plane, idx1: int, heading: Optional[float] = None, **kwargs) -> Tuple[bool, Dict]:
        """
        heading in radians
        """
        info = {}
        if idx1 != 0: # We only impose this constraint for edges connected to the starting waypoint (index=0)
            return True, info
        
        if heading is None:
            logger.warning(f"Yaw rate constraint is enabled but the own ship's heading was not provided (yaw rate constraint will be ignored).")
            return True, info

        # Compute edge speed
        dt = plane.get_time(*p2) - plane.get_time(*p1)

        if dt <= 0:
            return False, info
        
        # edge_speed = edge_length / dt
        edge_angle = atan2(p2[0] - p1[0], p2[1] - p1[1])
        angle_error = ssa(edge_angle - heading)
        yaw_rate_required = abs(angle_error) / dt
        is_valid = yaw_rate_required <= self.yaw_rate
        return is_valid, info # Warning is here because dt can be a numpy array in theory

class COLREGS(INodeFilter):
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