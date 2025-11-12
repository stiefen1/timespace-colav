"""
Typical use case:

"""

import networkx as nx
from typing import List, Tuple
from colav.obstacles.moving import MovingObstacle
from colav.path.pwl import PWLPath

class VisibilityGraph(nx.DiGraph):
    def __init__(
            self,
            p0: Tuple[float, float],
            pf: Tuple[float, float],
            obstacles: List[MovingObstacle]
    ):
        self.p0 = p0,
        self.pf = pf
        self.obstacles = obstacles
        super().__init__()
        self.populate_nodes()
        self.populate_edges()

    def populate_nodes(self) -> None:
        pass

    def populate_edges(self) -> None:
        pass

    def get_dijkstra_path(self) -> PWLPath:
        return PWLPath([])