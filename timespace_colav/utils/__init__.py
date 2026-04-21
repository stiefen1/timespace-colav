"""
Utility functions for maritime navigation and collision avoidance.

Provides mathematical functions for angle handling, coordinate transformations,
and MMSI (Maritime Mobile Service Identity) utilities for vessel identification.

Modules
-------
math : Mathematical utilities for navigation
    - Angle normalization and wrapping functions
    - Rotation matrix generation
    - Coordinate transformations

mmsi : MMSI generation and validation
    - Random and realistic MMSI generation
    - MMSI format validation
    - Maritime identification utilities
"""

__version__ = "0.1.0"

# Import main modules
from .math import *
from .mmsi import *