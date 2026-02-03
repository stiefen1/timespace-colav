"""
Timespace projection for maritime collision avoidance.

Provides tools for projecting moving obstacles into timespace footprints 
(static polygons) and creating constraint-based planning planes for 
collision-free navigation.

Key Components
--------------
Plane : Timespace coordinate plane for trajectory timing
TimeSpaceProjector : Projects moving obstacles into static polygons
SpeedConstraint : Speed limit enforcement for path planning
CourseRateConstraint : Course change rate limitations
COLREGS : Maritime collision regulations compliance

Examples
--------
>>> from colav.timespace import TimeSpaceProjector, Plane, COLREGS
>>> projector = TimeSpaceProjector(desired_speed=10.0)
>>> static_obstacles = projector.get(start, target, moving_ships)
>>> 
>>> # With constraints
>>> from colav.timespace import SpeedConstraint, CourseRateConstraint
>>> constraints = [SpeedConstraint(15.0), CourseRateConstraint(0.1)]
"""

__version__ = "0.1.0"

# Import main modules
from .plane import Plane
from .projector import TimeSpaceProjector
from .constraints import SpeedConstraint, CourseRateConstraint, COLREGS