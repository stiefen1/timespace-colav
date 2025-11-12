# from typing import List, Tuple
# from colav.obstacles.moving import MovingObstacle

# class StaticObstacle(MovingObstacle):
#     def __init__(
#         self,
#         geometry: List[ Tuple[float, float] ],
#         mmsi: int | None = None
#     ):
#         super().__init__(
#             position=(0, 0),
#             psi=0,
#             velocity=(0, 0),
#             geometry_at_psi_equal_0=geometry,
#             mmsi=mmsi
#         )