"""
COLREGS (International Regulations for Preventing Collisions at Sea) implementation.

Provides encounter classification and collision avoidance recommendations
based on maritime regulations for safe navigation in international waters.
Supports standard encounters and good seamanship practices.

Key Components
--------------
Encounter : Classification of vessel encounter types
Recommendation : Collision avoidance maneuver recommendations  
get_encounter : Classify encounter between two vessels
get_recommendation_for_os : Get COLREGS-compliant maneuver recommendation

Examples
--------
Basic encounter analysis:

>>> from colav.colregs import get_encounter, get_recommendation_for_os
>>> encounter = get_encounter(own_ship, target_ship)
>>> recommendation, info = get_recommendation_for_os(own_ship, target_ship)
>>> print(f"Encounter: {encounter.name}, Recommendation: {recommendation.name}")
"""

__version__ = "0.1.0"

# Import main modules
from .encounters import *