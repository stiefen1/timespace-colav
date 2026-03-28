"""
Path planning algorithms for collision avoidance.

Provides abstract interfaces and concrete implementations for finding
collision-free paths through static obstacles. Designed for integration
with timespace collision avoidance systems.

Key Classes
-----------
IPathPlanner : Abstract base class for path planners
PathPlanner : Basic path planner implementation  
VGPathPlanner : Visibility graph-based path planner

Notes
-----
All planners work with 2D spatial coordinates and static obstacles.
For dynamic obstacle avoidance, use in combination with timespace
projection methods.
"""
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Dict
from shapely import Polygon
from colav.path.graph import VisibilityGraph
from colav.path.filters import IEdgeFilter, INodeFilter
from colav.path.pwl import PWLPath
import networkx as nx, matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class IPathPlanner(ABC):
    """
    Abstract interface for maritime path planners.
    
    Defines the contract for path planning algorithms used in collision
    avoidance systems. All concrete planners must implement the core
    planning methods and visualization interface.
    
    Methods
    -------
    get(*args, **kwargs)
        Abstract method for computing collision-free path
    has_path(*args, **kwargs)
        Abstract method for checking path existence
    plot(ax=None, **kwargs)
        Abstract method for plotting the path planner state
        
    Notes
    -----
    Subclasses should implement specific path planning algorithms
    (A*, Dijkstra, RRT, etc.) while maintaining this interface.
    """
    def __init__(
            self
    ):
        pass

    @abstractmethod
    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Optional[Axes]:
        """
        Plot the path planner state (abstract method).
        
        Parameters
        ----------
        *args, **kwargs
            Planner-specific plotting arguments.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, may create new figure.
            
        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object containing the plot.
        """
        return ax
    
    @abstractmethod
    def get(self, *args, **kwargs) -> Optional[PWLPath]:
        """
        Compute collision-free path (abstract method).
        
        Parameters
        ----------
        *args, **kwargs
            Planner-specific arguments for path computation.
            
        Returns
        -------
        PWLPath or None
            Computed path or None if no path exists.
        """
        pass

    @abstractmethod
    def has_path(self, *args, **kwargs) -> bool:
        """
        Check if collision-free path exists (abstract method).
        
        Parameters
        ----------
        *args, **kwargs
            Planner-specific arguments for connectivity check.
            
        Returns
        -------
        bool
            True if path exists, False otherwise.
        """
        pass
    
class PathPlanner(IPathPlanner):
    """
    Basic path planner implementation.
    
    Simple concrete implementation of IPathPlanner interface.
    Currently provides minimal functionality - extend for specific
    path planning algorithms.
    
    Methods
    -------
    plot(ax=None, **kwargs)
        Basic plotting implementation (currently returns input axes)
        
    Notes
    -----
    This is a placeholder implementation. For actual path planning,
    use VGPathPlanner or implement custom algorithms.
    """
    def __init__(
            self
    ):
        pass

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Optional[Axes]:
        """
        Basic plot implementation.
        
        Parameters
        ----------
        *args, **kwargs
            Unused plotting arguments.
        ax : matplotlib.axes.Axes, optional
            Axes to return unchanged.
            
        Returns
        -------
        matplotlib.axes.Axes or None
            Input axes object.
        """
        return ax
    
    def get(self, *args, **kwargs) -> Optional[PWLPath]:
        """
        Basic get implementation (returns None).
        
        Parameters
        ----------
        *args, **kwargs
            Unused arguments.
            
        Returns
        -------
        None
            Always returns None as no path planning is implemented.
        """
        return None

    def has_path(self, *args, **kwargs) -> bool:
        """
        Basic has_path implementation (returns False).
        
        Parameters
        ----------
        *args, **kwargs
            Unused arguments.
            
        Returns
        -------
        bool
            Always returns False as no path planning is implemented.
        """
        return False
    

class VGPathPlanner(IPathPlanner, VisibilityGraph):
    """
    Visibility graph-based path planner for maritime navigation.
    
    Constructs a visibility graph connecting start/end points through
    obstacle vertices, then finds shortest collision-free paths using
    Dijkstra's algorithm. Optimal for polygonal static obstacles.
    
    Parameters
    ----------
    p_0 : tuple of float
        Start position (x, y) in meters.
    p_f : tuple of float
        Target position (x, y) in meters.
    obstacles : dict of int to Polygon, optional
        Static obstacles as shapely Polygon objects.
    edge_filters : list of IEdgeFilter, optional
        Filters to exclude certain edges from graph.
    node_filters : list of INodeFilter, optional
        Filters to exclude certain nodes from graph.
    **kwargs
        Additional arguments passed to VisibilityGraph.
        
    Methods
    -------
    has_path()
        Check if path exists from start to target
    get()
        Find shortest path using Dijkstra's algorithm
    plot(ax=None, **kwargs)
        Visualize the visibility graph
        
    Examples
    --------
    >>> from shapely import Polygon
    >>> obstacles = {0: Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])}
    >>> planner = VGPathPlanner((0, 0), (30, 30), obstacles)
    >>> if planner.has_path():
    >>>     path = planner.get()
    >>>     print(f"Path length: {path.length:.1f}m")
    
    Notes
    -----
    - Assumes polygonal obstacles with vertex-to-vertex visibility
    - Optimal for sparse obstacle environments
    - For dense environments, consider sampling-based methods (RRT, PRM)
    - Integrates with VisibilityGraph for geometric computations
    """
    def __init__(
        self,
        p_0: Tuple[float, float],
        p_f: Tuple[float, float],
        obstacles: Optional[Dict[int, Polygon]] = None,
        edge_filters: Optional[List[IEdgeFilter]] = None,
        node_filters: Optional[List[INodeFilter]] = None,
        **kwargs
    ):
        IPathPlanner.__init__(self)
        VisibilityGraph.__init__(
            self,
            p_0,
            p_f,
            obstacles,
            edge_filters,
            node_filters,
            **kwargs
        )

    def has_path(self) -> bool:
        """
        Check if collision-free path exists from start to target.
        
        Returns
        -------
        bool
            True if path exists, False otherwise.
            
        Notes
        -----
        Uses NetworkX path existence check on visibility graph.
        Fast connectivity test before expensive path computation.
        """
        return nx.has_path(self, source=0, target=-1)

    def get(self) -> PWLPath:
        """
        Find shortest collision-free path using Dijkstra's algorithm.
        
        Computes optimal path through visibility graph from start to target.
        Guarantees shortest distance while avoiding obstacles.
        
        Returns
        -------
        PWLPath
            Piecewise linear path with waypoints from start to target.
            
        Raises
        ------
        NetworkXNoPath
            If no path exists between start and target.
            
        Notes
        -----
        Path optimality depends on visibility graph completeness.
        Use has_path() first to check connectivity.
        """
        path_nodes = nx.dijkstra_path(self, source=0, target=-1, weight='weight')
        waypoints = [self.nodes[node]["pos"] for node in path_nodes]
        return PWLPath(waypoints)

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """
        Plot the visibility graph for visualization.
        
        Displays nodes, edges, and connectivity of the visibility graph.
        Useful for debugging path planning and understanding obstacle
        navigation strategies.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments passed to networkx.draw() for styling.
            Common options: node_color, edge_color, node_size.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates new figure if None.
            
        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the visibility graph plot.
            
        Examples
        --------
        >>> planner.plot(node_color='red', node_size=50)
        >>> planner.plot(ax=existing_axes, edge_color='blue')
        """
        if ax is None:
            _, ax = plt.subplots()
        
        positions = nx.get_node_attributes(self, "pos")
        nx.draw(self, pos=positions, ax=ax, *args, **kwargs)
        return ax


if __name__ == "__main__":
    from shapely import Polygon
    import matplotlib.pyplot as plt
    
    # Simple example
    obstacles = {
        1: Polygon([(15, 10), (25, 10), (25, 20), (15, 20)]),
        2: Polygon([(20, 25), (30, 25), (30, 35), (20, 35)])
    }
    planner = VGPathPlanner((0, 0), (45, 45), obstacles)
    
    if planner.has_path():
        path = planner.get()
        print(f"Path length: {path.length:.1f}m")
        
        # Plot
        fig, ax = plt.subplots()
        for obs in obstacles.values():
            ax.fill(*obs.exterior.xy, 'red', alpha=0.5)
        path.plot('--b', ax=ax, linewidth=2)
        ax.plot(0, 0, 'go', markersize=8)  # start
        ax.plot(45, 45, 'ro', markersize=8)  # target
        ax.set_aspect('equal')
        plt.show()
    else:
        print("No path found")