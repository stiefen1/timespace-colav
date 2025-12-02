from colav.path.edge_filter import IEdgeFilter
from colav.timespace.plane import Plane
import networkx as nx
from typing import Dict, Tuple
from math import atan2
from colav.utils.math import ssa

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

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], plane: Plane, heading: float, idx1: int, **kwargs) -> Tuple[bool, Dict]:
        """
        heading in radians
        """
        if idx1 != 0: # We only impose this constraint for edges connected to the current waypoint
            return True, {}

        # Compute edge speed
        info = {}
        dt = plane.get_time(*p2) - plane.get_time(*p1)

        if dt <= 0:
            return False, info
        
        # edge_speed = edge_length / dt
        edge_angle = atan2(p2[0] - p1[0], p2[1] - p1[1])
        angle_error = ssa(edge_angle - heading)
        yaw_rate_required = abs(angle_error) / dt
        is_valid = yaw_rate_required <= self.yaw_rate
        return is_valid, info # Warning is here because dt can be a numpy array in theory
    
class COLREGS(IEdgeFilter):
    def __init__(
        self,
        good_seamanship: bool = False
    ):
        super().__init__()
        self.good_seamanship = good_seamanship

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], graph: nx.DiGraph, plane: Plane) -> Tuple[bool, Dict]:
        info = {}
        return True, info