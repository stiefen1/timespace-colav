"""
Visibility graph implementation for maritime path planning.

Constructs graphs connecting obstacle vertices and waypoints through
line-of-sight edges for optimal collision-free path computation.
Essential component for geometric path planning in obstacle environments.

Key Components
--------------
is_edge_visible : function
    Check collision-free visibility between points
relocate_colliding_point : function  
    Move points outside obstacle boundaries
VisibilityGraph : class
    NetworkX-based visibility graph for path planning

Notes
-----
Designed for polygonal static obstacles in maritime navigation.
Integrates with filter systems for constraint enforcement.
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict
from colav.path.filters import IEdgeFilter, INodeFilter
from shapely import Polygon, Point, LineString, MultiPoint 
import matplotlib.pyplot as plt, logging
logger = logging.getLogger(__name__)

NUMBER_OF_NODES_BEFORE_WARNING = 100

def is_edge_visible(edge: LineString, obstacles: List[Polygon]) -> bool:
    """
    Check if edge is collision-free among obstacles.
    
    Tests whether a straight line segment intersects with any obstacle
    interior. Essential for visibility graph construction.
    
    Parameters
    ----------
    edge : LineString
        Line segment to test for collisions.
    obstacles : list of Polygon
        Static obstacle geometries to check against.
        
    Returns
    -------
    bool
        True if edge is collision-free, False if intersecting obstacles.
        
    Notes
    -----
    Uses Shapely geometric predicates for robust collision detection.
    Distinguishes between crossing (collision) and touching (boundary contact).
    """
    for polygon in obstacles:
        if (polygon.crosses(edge) and not polygon.touches(edge)) or edge.within(polygon):
            return False
    return True


def relocate_colliding_point(
    p_coll: Tuple[float, float], 
    p_target: Tuple[float, float], 
    obstacles: List[Polygon], 
    max_iter: int = 10, 
    buffer_distance: float = 1e-3
) -> Tuple[float, float]:
    """
    Relocate point outside obstacles using iterative projection.
    
    Moves colliding points to obstacle boundaries along the line toward
    target. Used to ensure start/end points are in collision-free space.
    
    Parameters
    ----------
    p_coll : tuple of float
        Point coordinates (x, y) that may be inside obstacles.
    p_target : tuple of float
        Target point coordinates (x, y) to move toward.
    obstacles : list of Polygon
        Obstacle geometries to avoid.
    max_iter : int, default 10
        Maximum relocation iterations before giving up.
    buffer_distance : float, default 1e-3
        Small buffer distance from obstacle boundary.
        
    Returns
    -------
    tuple of float
        Relocated point coordinates (x, y).
        
    Notes
    -----
    Uses line-obstacle intersection to find boundary points.
    Handles complex geometry types (Point, MultiPoint, LineString).
    May fail if both points are inside obstacles.
    """
    for i in range(max_iter):
        colliding = False
        for obs in obstacles:
            if obs.contains(Point(p_coll)):
                logger.warning(f"{p_coll} is colliding with an obstacle, relocating (iteration {i+1}/{max_iter})")
                colliding = True
                line_from_p_coll_to_p_target = LineString([p_coll, p_target])
                buffered_obstacle = Polygon(obs.buffer(buffer_distance).exterior.coords)

                new_p_coll = line_from_p_coll_to_p_target.intersection(buffered_obstacle.exterior)
                if isinstance(new_p_coll, MultiPoint):
                    new_p_coll = new_p_coll.geoms[-1]
                elif isinstance(new_p_coll, LineString) and not(new_p_coll.is_empty):
                    # xy = tuple(new_p_coll.coords.xy[0]) -> old version, before claude told me it was wrong. Was it ?
                    # new_p_coll = Point(xy[0], xy[1])
                    coords = list(new_p_coll.coords)
                    new_p_coll = Point(coords[0])
                elif not(isinstance(new_p_coll, Point)): 
                    logger.warning(f"Failed to relocate colliding point. type={type(new_p_coll)}") # This can happen if both p_coll and p_target are inside an obstacle
                    return p_coll
                p_coll = (new_p_coll.x, new_p_coll.y)
                break
        if not colliding:
            logger.info(f"Successfully relocated point to {p_coll} in {i} iterations.")
            return p_coll
    return p_coll


class VisibilityGraph(nx.DiGraph):
    """
    Visibility graph for obstacle-aware path planning.
    
    Constructs a directed graph where nodes represent obstacle vertices
    plus start/end waypoints, connected by collision-free visibility edges.
    Enables optimal geometric path planning using graph algorithms.
    
    Parameters
    ----------
    p_0 : tuple of float
        Start point coordinates (x, y) in meters.
    p_f : tuple of float
        Target point coordinates (x, y) in meters.
    obstacles : dict of int to Polygon, optional
        Mapping from obstacle IDs to Polygon geometries.
    edge_filters : list of IEdgeFilter, optional
        Constraint filters applied to potential edges.
    node_filters : list of INodeFilter, optional
        Constraint filters applied to obstacle vertices.
    **kwargs
        Additional arguments passed to filter functions.
        
    Attributes
    ----------
    p_0, p_f : tuple of float
        Start and target point coordinates (may be relocated).
    obstacles : dict
        Obstacle ID to Polygon mapping.
    edge_filters, node_filters : list
        Applied constraint filters.
        
    Methods
    -------
    populate_nodes(**kwargs)
        Add all graph nodes (waypoints + obstacle vertices)
    populate_edges(**kwargs)
        Add collision-free visibility edges between nodes
        
    Examples
    --------
    Basic visibility graph construction:
    
    >>> from shapely import Polygon
    >>> obstacles = {1: Polygon([(0,0), (10,0), (10,10), (0,10)])}
    >>> vg = VisibilityGraph((-5,-5), (15,15), obstacles)
    >>> print(f"Graph has {len(vg.nodes)} nodes, {len(vg.edges)} edges")
    
    With custom filters:
    
    >>> from colav.path.filters import AngleFilter
    >>> vg = VisibilityGraph(
    ...     (-5,-5), (15,15), obstacles,
    ...     edge_filters=[AngleFilter()],
    ...     max_angle=45  # Passed to filters
    ... )
    
    Notes
    -----
    - Inherits from NetworkX DiGraph for standard graph algorithms
    - Node IDs: 0=start, positive=obstacle vertices, -1=target
    - Edge weights set to Euclidean distances
    - Computational complexity: O(n²) for n vertices
    - Use obstacle simplification for complex polygons (>100 vertices)
    
    See Also
    --------
    VGPathPlanner : Higher-level path planning interface
    colav.path.filters : Constraint filter implementations
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
        """
        Initialize visibility graph with obstacles and constraints.
        
        Constructs complete visibility graph by adding all nodes
        (waypoints + obstacle vertices) and connecting visible pairs.
        
        Parameters
        ----------
        p_0 : tuple of float
            Start waypoint coordinates (x, y).
        p_f : tuple of float  
            Target waypoint coordinates (x, y).
        obstacles : dict of int to Polygon, optional
            Static obstacles mapped by unique IDs.
        edge_filters : list of IEdgeFilter, optional
            Filters to constrain allowable edges.
        node_filters : list of INodeFilter, optional
            Filters to constrain allowable nodes.
        **kwargs
            Additional arguments passed to populate methods and filters.
            
        Notes
        -----
        Graph construction happens during initialization.
        Start/target points are automatically relocated if inside obstacles.
        """
        self.p_0 = p_0
        self.p_f = p_f
        self.obstacles = obstacles or {}
        self.edge_filters = edge_filters or []
        self.node_filters = node_filters or []

        # Initialize graph and populate with nodes and edges
        super().__init__()
        self.populate_nodes(**kwargs)
        self.populate_edges(**kwargs)
        logger.debug(f"Succesfully created VisibilityGraph.")

    def populate_nodes(self, relocation_buffer_distance: float = 1e-3, **kwargs) -> None:
        """
        Add all graph nodes: waypoints and obstacle vertices.
        
        Creates nodes for start point (ID=0), all obstacle vertices
        (positive IDs), and target point (ID=-1). Applies node filters
        and relocates waypoints if they collide with obstacles.
        
        Parameters
        ----------
        relocation_buffer_distance : float, default 1e-3
            Buffer distance for obstacle boundary relocation.
        **kwargs
            Additional arguments passed to node filters.
            
        Notes
        -----
        - Node numbering: 0=start, 1,2,3...=vertices, -1=target
        - Skips first polygon vertex (duplicate of last)
        - Issues warning if graph exceeds 100 nodes (performance concern)
        - Node attributes include position, obstacle ID, and filter results
        """
        node_idx = 1

        # Ensure start and end points are not inside obstacles
        obstacles_list = list(self.obstacles.values())
        logger.debug(f"Received {len(obstacles_list)} obstacles, start populating nodes.")

        for obs in obstacles_list:
            if obs.contains(Point(self.p_f)):
                self.p_f = relocate_colliding_point(
                    self.p_f, self.p_0, obstacles_list, buffer_distance=relocation_buffer_distance
                )
            if obs.contains(Point(self.p_0)):
                self.p_0 = relocate_colliding_point(
                    self.p_0, self.p_f, obstacles_list, buffer_distance=relocation_buffer_distance
                )

        # Add start node
        self.add_node(0, pos=self.p_0, id=0, label='start')
        
        # Add obstacle vertex nodes
        for obstacle_id, polygon in self.obstacles.items():
            assert obstacle_id != -1 and obstacle_id != 0, f"Obstacle ID must be != -1, 0. Got {obstacle_id}"

            # Add each vertex of the polygon (skip first as it equals last)
            for i, (x, y) in enumerate(zip(*polygon.exterior.xy)):
                if i == 0:  # Skip duplicate first vertex
                    continue
                node = {
                    'pos': (x, y),
                    'id': obstacle_id,
                }

                centroid = polygon.centroid

                all_valid = True
                info = {}
                for node_filter in self.node_filters:
                    valid, info = node_filter(node=node, p_0=self.p_0, obstacle_centroid=(centroid.x, centroid.y), **kwargs)
                    all_valid = all_valid and valid
                    if not(all_valid):
                        break

                if all_valid:
                    self.add_node(node_idx, pos=(x, y), id=obstacle_id, **info)
                    node_idx += 1
        
        # Add end node
        self.add_node(-1, pos=self.p_f, id=-1, label='end')
        logger.debug(f"Successfully populated {node_idx} nodes.")

        if node_idx > NUMBER_OF_NODES_BEFORE_WARNING:
            logger.warning(f"Graph contains {node_idx} > {NUMBER_OF_NODES_BEFORE_WARNING} nodes, potentially leading to high computational demand. Try to simplify the obstacles shape using shapely.simplify() method.")

    def populate_edges(self, **kwargs) -> None:
        """
        Connect all mutually visible node pairs with weighted edges.
        
        Tests line-of-sight visibility between every node pair,
        applies edge filters, and adds collision-free connections
        with Euclidean distance weights.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to edge filters.
            
        Notes
        -----
        - Computational complexity: O(n²) for n nodes
        - Edge weights represent straight-line distances in meters
        - Visibility test performed before expensive filter evaluation
        - Self-loops automatically excluded (node_1 != node_2)
        - Failed filter results immediately break evaluation chain
        """
        obstacles_list = list(self.obstacles.values())

        logger.debug(f"Received {len(obstacles_list)} obstacles, start populating edges.")
        
        for node_1 in self.nodes:
            for node_2 in self.nodes:
                if node_1 != node_2:
                    # Create edge as line segment
                    edge_line = LineString([
                        self.nodes[node_1]["pos"], 
                        self.nodes[node_2]["pos"]
                    ])

                    # Avoid using edge filters if edge is not even visible
                    if not(is_edge_visible(edge_line, obstacles_list)):
                        continue

                    # Apply all edge filters
                    valid = True
                    for edge_filter in self.edge_filters:
                        valid_edge, info = edge_filter(
                            self.nodes[node_1],
                            self.nodes[node_2],
                            graph=self.graph,
                            idx1=node_1,
                            **kwargs
                        )

                        valid = valid and valid_edge
                        if not valid: # immediately break for loop if not valid
                            break

                    # Add edge if it passes filters and is collision-free
                    if valid:
                        self.add_edge(node_1, node_2, weight=edge_line.length)

        logger.debug(f"Graph was successfully populated.")
    
if __name__ == "__main__":
    # Example usage: path planning around two ship obstacles
    from colav.obstacles import MovingShip
    from colav.path.planning import VGPathPlanner
    
    # Create two ship obstacles
    ship1 = MovingShip((0, 0), 30, (4, 3), 10, 3, degrees=True, mmsi=111)
    ship2 = MovingShip((10, 10), -120, (-2, -3), 5, 2, degrees=True, mmsi=222)
    
    # Convert to static polygon obstacles
    obs1 = Polygon(ship1.geometry)
    obs2 = Polygon(ship2.geometry)
    
    # Build visibility graph
    planner = VGPathPlanner(
        p_0=(-5, -5),          # Start point
        p_f=(15, 15),          # End point  
        obstacles={111: obs1, 222: obs2},
        edge_filters=[],
        max_angle=-30
    )
    
    # Create clean visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot visibility graph with clean styling
    planner.plot(ax=ax, with_labels=True, node_color='#2a9d8f', 
                node_size=200, edge_color='#264653')
    
    # Plot obstacles with maritime colors
    ship1.fill(ax=ax, c='#e63946', alpha=0.7, label='Ship 1')
    ship2.fill(ax=ax, c='#e63946', alpha=0.7, label='Ship 2')
    
    # Plot optimal path
    optimal_path = planner.get()
    optimal_path.plot(ax=ax, c='#f77f00', linewidth=3, label='Optimal Path')
    
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, color='#dee2e6')
    ax.set_title('Visibility Graph Path Planning', fontsize=14, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

