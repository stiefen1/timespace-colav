"""
Standard geometric shapes for obstacle representation.

Provides factory functions for creating common obstacle geometries
used in maritime collision avoidance, including ships, circular
obstacles, and elliptical safety zones.

Key Functions
-------------
SHIP : Maritime vessel outline based on length and beam
CIRCLE : Circular obstacle with configurable radius
ELLIPSE : Elliptical obstacle with independent axes

Notes
-----
All shapes are defined with origin at (0,0) and can be transformed
using the transform module functions. Shapes return lists of (x,y)
coordinate tuples suitable for polygon creation.

Examples
--------
>>> ship_outline = SHIP(loa=20, beam=5)
>>> circular_zone = CIRCLE(radius=50)
>>> safety_ellipse = ELLIPSE(ax_surge=30, ax_sway=15)
"""

from shapely import Point, affinity, simplify

SHIP = lambda loa, beam: [
    (0, loa/2),
    (beam/2, loa/4),
    (beam/2, -loa/2),
    (0, -loa/2),
    (-beam/2, -loa/2),
    (-beam/2, loa/4),
    (0, loa/2)
]
"""Create ship-shaped outline.

Parameters:
    loa (float): Length overall in meters
    beam (float): Beam (width) in meters
    
Returns:
    list: Ship outline coordinates with bow pointing north
    
Examples:
    >>> ship = SHIP(20, 5)  # 20m long, 5m wide ship
"""

CIRCLE = lambda r, tol=None: [*zip(*simplify(Point(0, 0).buffer(r), tol or r/15).exterior.coords.xy)]
"""Create circular obstacle outline.

Parameters:
    r (float): Radius in meters
    tol (float, optional): Simplification tolerance
    
Returns:
    list: Circle outline coordinates centered at origin
    
Examples:
    >>> circle = CIRCLE(25)  # 25m radius circular zone
"""

ELLIPSE = lambda ax_surge, ax_sway, tol=None: [*zip(*simplify(affinity.scale(Point(0, 0).buffer(1), ax_sway, ax_surge), tol or (ax_surge+ax_sway)/30).exterior.coords.xy)]
"""Create elliptical obstacle outline.

Parameters:
    ax_surge (float): Semi-axis along surge direction (length)
    ax_sway (float): Semi-axis along sway direction (width)  
    tol (float, optional): Simplification tolerance
    
Returns:
    list: Ellipse outline coordinates centered at origin
    
Examples:
    >>> ellipse = ELLIPSE(30, 15)  # 30m x 15m elliptical safety zone
"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print(len(CIRCLE(2)))
    plt.figure()
    plt.plot(*zip(*CIRCLE(2)))
    plt.plot(*zip(*ELLIPSE(2, 1)))
    plt.plot(*zip(*SHIP(2, 1)))
    plt.show()