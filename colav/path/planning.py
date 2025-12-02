"""
Implement base class for path planner to be used within this colav framework

"""
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Dict
from shapely import Polygon
from colav.path.graph import VisibilityGraph
from colav.path.edge_filter import IEdgeFilter
from colav.path.pwl import PWLPath
import networkx as nx, matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class IPathPlanner(ABC):
    def __init__(
            self
    ):
        pass

    @abstractmethod
    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Optional[Axes]:
        return ax
    
class PathPlanner(IPathPlanner):
    def __init__(
            self
    ):
        pass

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Optional[Axes]:
        return ax
    

class VGPathPlanner(IPathPlanner, VisibilityGraph):
    def __init__(
        self,
        p_0: Tuple[float, float],
        p_f: Tuple[float, float],
        obstacles: Optional[Dict[int, Polygon]] = None,
        edge_filters: Optional[List[IEdgeFilter]] = None,
        **kwargs
    ):
        IPathPlanner.__init__(self)
        VisibilityGraph.__init__(
            self,
            p_0,
            p_f,
            obstacles,
            edge_filters,
            **kwargs
        )

    def has_path(self) -> bool:
        return nx.has_path(self, source=0, target=-1)

    def get_dijkstra_path(self) -> PWLPath:
        """
        Find shortest collision-free path using Dijkstra's algorithm.
        
        Returns:
            PWLPath object containing waypoints from start to end
        """
        path_nodes = nx.dijkstra_path(self, source=0, target=-1, weight='weight')
        waypoints = [self.nodes[node]["pos"] for node in path_nodes]
        return PWLPath(waypoints)

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """
        Plot the visibility graph.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            *args, **kwargs: Additional arguments passed to networkx.draw()
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            _, ax = plt.subplots()
        
        positions = nx.get_node_attributes(self, "pos")
        nx.draw(self, pos=positions, ax=ax, *args, **kwargs)
        return ax