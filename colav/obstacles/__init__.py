"""
Obstacle representation and geometry for maritime collision avoidance.

Provides classes and utilities for representing moving and static obstacles
in maritime environments, including vessel dynamics, geometric shapes,
and spatial transformations for collision detection and avoidance.

Key Components
--------------
MovingObstacle : Generic moving obstacle with position and velocity
MovingShip : Maritime vessel with ship-specific dynamics and uncertainty
SHIP, CIRCLE, ELLIPSE : Common obstacle shape definitions
translate, rotate : Geometric transformation functions

Examples
--------
Basic moving ship:

>>> from colav.obstacles import MovingShip
>>> ship = MovingShip.from_body(
...     position=(100, 50), psi=90, u=5, v=0,
...     loa=20, beam=5, degrees=True, mmsi=123456789
... )
>>> future_ship = ship.predict(dt=10)  # Position after 10 seconds
"""

__version__ = "0.1.0"

# Import main modules
from .shapes import *
from .moving import *
from .transform import *