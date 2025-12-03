from abc import ABC, abstractmethod
from typing import Dict, Tuple

class IEdgeFilter(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], *args, **kwargs) -> Tuple[bool, Dict]:
        info = {}
        return True, info

    def __call__(self, n1: Dict, n2: Dict, *args, **kwargs) -> Tuple[bool, Dict]:
        return self.is_valid(n1["pos"], n2["pos"], *args, **kwargs)
    
class MaxAngleEdgeFilter(IEdgeFilter):
    """
    Example for using the IEdgeFilter base class
    """
    def __init__(
            self,
            angle_max_deg: float = 30
    ):
        self.angle_max_deg = angle_max_deg

    def is_valid(self, p1: Tuple[float, float], p2: Tuple[float, float], *args, **kwargs) -> Tuple[bool, Dict]:
        info = {}
        return True, info
    
class INodeFilter(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def is_valid(self, node: Dict, *args, **kwargs) -> Tuple[bool, Dict]:
        info = {}
        return True, info

    def __call__(self, node: Dict, *args, **kwargs) -> Tuple[bool, Dict]:
        return self.is_valid(node, *args, **kwargs)


