"""
Visibility graph implementation for path planning with obstacles.

Creates a graph where nodes are obstacle vertices and start/end points,
and edges connect visible (collision-free) node pairs.
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict
from colav.path.filters import IEdgeFilter, INodeFilter
from colav.obstacles.moving import MovingObstacle
from shapely import Polygon, Point, LineString, MultiPoint, MultiPolygon
import matplotlib.pyplot as plt, logging
logger = logging.getLogger(__name__)


def is_edge_visible(edge: LineString, obstacles: List[Polygon]) -> bool:
    """
    Check if an edge (line segment) is visible (collision-free) among obstacles.
    
    Args:
        edge: LineString representing the edge to check
        obstacles: List of polygon obstacles
        
    Returns:
        True if edge doesn't intersect any obstacle interior
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
    Move a point outside obstacles if it's colliding.
    
    Iteratively moves the point along the line toward the target until
    it's no longer inside any obstacle.
    
    Args:
        p_coll: Point that may be colliding with obstacles
        p_target: Target point to move toward
        obstacles: List of polygon obstacles to avoid
        max_iter: Maximum relocation attempts
        buffer_distance: Small buffer distance for obstacle boundary
        
    Returns:
        Relocated point coordinates
    """
    for i in range(max_iter):
        colliding = False
        for obs in obstacles:
            if obs.contains(Point(p_coll)):
                logger.warning(f"{p_coll} is colliding with an obstacle, relocating (iteration {i+1}/{max_iter})")
                colliding = True
                line_from_p_coll_to_p_target = LineString([p_coll, p_target])
                buffered_obstacle = obs.buffer(buffer_distance)

                if isinstance(buffered_obstacle, MultiPolygon):
                    buffered_obstacle = buffered_obstacle.convex_hull

                new_p_coll = line_from_p_coll_to_p_target.intersection(buffered_obstacle.exterior)
                if isinstance(new_p_coll, MultiPoint):
                    new_p_coll = new_p_coll.geoms[-1]
                assert isinstance(new_p_coll, Point), "Failed to relocate colliding point"
                p_coll = (new_p_coll.x, new_p_coll.y)
                break
        if not colliding:
            logger.info(f"Successfully relocated point to {p_coll} in {i} iterations.")
            return p_coll
    return p_coll


class VisibilityGraph(nx.DiGraph):
    """
    Visibility graph for path planning in obstacle environments.
    
    Builds a directed graph where nodes are obstacle vertices plus start/end points,
    and edges connect mutually visible (collision-free) node pairs.
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
        Initialize visibility graph.
        
        Args:
            p_0: Start point coordinates
            p_f: End point coordinates  
            obstacles: Dictionary mapping obstacle IDs to Polygon objects
            edge_filters: List of functions to filter allowable edges
            **kwargs: Additional arguments passed to node/edge population
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
        Add all nodes to the graph: start point, obstacle vertices, end point.
        
        Relocates start/end points if they're inside obstacles.
        Node IDs: 0 = start, positive = obstacle vertices, -1 = end
        
        Args:
            relocation_buffer_distance: Buffer distance for obstacle boundary relocation
        """
        node_idx = 1

        # Ensure start and end points are not inside obstacles
        obstacles_list = list(self.obstacles.values())
        logger.debug(f"Received {len(obstacles_list)} obstacles, start populating nodes.")

        for obs in obstacles_list:
            if obs.contains(Point(self.p_0)):
                self.p_0 = relocate_colliding_point(
                    self.p_0, self.p_f, obstacles_list, buffer_distance=relocation_buffer_distance
                )
            if obs.contains(Point(self.p_f)):
                self.p_f = relocate_colliding_point(
                    self.p_f, self.p_0, obstacles_list, buffer_distance=relocation_buffer_distance
                )

        # Add start node
        self.add_node(0, pos=self.p_0, id=0, label='start')
        
        # Add obstacle vertex nodes
        i: int = 0
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

        if i > 100:
            logger.warning(f"Graph contains {node_idx} > 100 nodes, potentially leading to high computational demand. Try to simplify the obstacles shape using shapely.simplify() method.")

    def populate_edges(self, **kwargs) -> None:
        """
        Add edges between all mutually visible node pairs.
        
        Tests visibility between every pair of nodes and adds edges for
        collision-free connections that pass all filter criteria.
        Edge weights are set to Euclidean distance.
        
        Args:
            **kwargs: Additional arguments passed to edge filters
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
    
    # Plot results
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Plot graph, obstacles, and optimal path
    planner.plot(ax=ax, with_labels=True, node_color='lightblue', node_size=300)
    ship1.fill(ax=ax, c='red', alpha=0.5, label='Ship 1')
    ship2.fill(ax=ax, c='red', alpha=0.5, label='Ship 2')
    
    # Plot shortest path
    optimal_path = planner.get_dijkstra_path()
    optimal_path.plot(ax=ax, c='blue', linewidth=3, label='Optimal Path')
    
    ax.legend()
    ax.set_title('Visibility Graph Path Planning')
    ax.grid(True, alpha=0.3)
    plt.show()

