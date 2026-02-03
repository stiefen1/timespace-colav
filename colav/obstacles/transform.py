"""
Geometric transformations for obstacle shapes and coordinates.

Provides functions for translating, rotating, and positioning geometric
shapes in 2D space for obstacle representation and collision detection
in maritime navigation scenarios.

Key Functions
-------------
translate : Move shape by displacement vector
rotate : Rotate shape around specified point
get_shape_at_xypsi : Position and orient shape at given location

Notes
-----
All functions work with shapes represented as lists of (x,y) coordinate
tuples. Rotations follow maritime convention with positive angles
representing clockwise rotation from north.

Examples
--------
>>> from colav.obstacles import SHIP
>>> ship_shape = SHIP(20, 5)
>>> positioned_ship = get_shape_at_xypsi(
...     x=100, y=50, psi=45, shape=ship_shape, degrees=True
... )
"""

from typing import List, Tuple
from colav.utils.math import rotation_matrix
import numpy as np

def translate(shape: List[ Tuple[float, float] ], dx: float = 0, dy: float = 0) -> List[ Tuple[float, float]]:
    """
    Translate shape by displacement vector.
    
    Parameters
    ----------
    shape : list of tuple
        Shape coordinates as (x, y) tuples.
    dx : float, default 0
        Displacement in x direction (east).
    dy : float, default 0
        Displacement in y direction (north).
        
    Returns
    -------
    list of tuple
        Translated shape coordinates.
        
    Examples
    --------
    >>> translated = translate(ship_shape, dx=100, dy=50)
    """
    return (np.array(shape) + np.array([dx, dy])).tolist()

def rotate(shape: List[ Tuple[float, float] ], angle: float, x: float = 0, y: float = 0, degrees: bool = False) -> List[ Tuple[float, float]]:
    """
    Rotate shape around specified point.
    
    Parameters
    ----------
    shape : list of tuple
        Shape coordinates as (x, y) tuples.
    angle : float
        Rotation angle. Positive values rotate clockwise.
    x, y : float, default 0
        Center of rotation coordinates.
    degrees : bool, default False
        Whether angle is in degrees (True) or radians (False).
        
    Returns
    -------
    list of tuple
        Rotated shape coordinates.
        
    Examples
    --------
    >>> rotated = rotate(ship_shape, 45, degrees=True)
    >>> rotated_around_point = rotate(shape, 30, x=100, y=50, degrees=True)
    """
    translated_shape = np.array(translate(shape, dx=-x, dy=-y))
    return translate((rotation_matrix(np.deg2rad(-angle) if degrees else -angle)[0:2, 0:2] @ translated_shape.T).T.tolist(), x, y)

def get_shape_at_xypsi(x: float, y: float, psi: float, shape: List[ Tuple[float, float]], degrees: bool = False) -> List[ Tuple[float, float]]:
    """
    Position and orient shape at specified location and heading.
    
    Combines translation and rotation to place a shape at the given
    position with the specified heading angle.
    
    Parameters
    ----------
    x, y : float
        Target position coordinates in meters.
    psi : float
        Heading angle. Positive clockwise from north.
    shape : list of tuple
        Shape coordinates as (x, y) tuples.
    degrees : bool, default False
        Whether psi is in degrees (True) or radians (False).
        
    Returns
    -------
    list of tuple
        Transformed shape coordinates at target position and heading.
        
    Examples
    --------
    >>> from colav.obstacles import SHIP
    >>> ship_at_pos = get_shape_at_xypsi(
    ...     x=100, y=50, psi=45, 
    ...     shape=SHIP(20, 5), degrees=True
    ... )
    """
    return rotate(translate(shape, dx=x, dy=y), psi, x=x, y=y, degrees=degrees)

if __name__ == "__main__":
    """
    This example was AI-generated using Claude Sonnet 4.
    """
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
    import numpy as np
    from colav.obstacles import SHIP

    # Initial parameters
    initial_loa = 2.0
    initial_beam = 1.0
    initial_x = 0.0
    initial_y = 0.0
    initial_psi = 0.0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.35)

    # Create initial ship shape
    ship_shape = SHIP(initial_loa, initial_beam)
    transformed_shape = get_shape_at_xypsi(initial_x, initial_y, initial_psi, ship_shape, degrees=True)

    # Plot initial ship
    line, = ax.plot(*zip(*transformed_shape), 'b-', linewidth=2, marker='o')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Interactive Ship Geometry')

    # Create sliders
    ax_loa = plt.axes((0.1, 0.25, 0.35, 0.03))
    ax_beam = plt.axes((0.55, 0.25, 0.35, 0.03))
    ax_x = plt.axes((0.1, 0.20, 0.35, 0.03))
    ax_y = plt.axes((0.55, 0.20, 0.35, 0.03))
    ax_psi = plt.axes((0.1, 0.15, 0.8, 0.03))

    slider_loa = widgets.Slider(ax_loa, 'LOA', 0.5, 5.0, valinit=initial_loa)
    slider_beam = widgets.Slider(ax_beam, 'Beam', 0.2, 3.0, valinit=initial_beam)
    slider_x = widgets.Slider(ax_x, 'X', -4.0, 4.0, valinit=initial_x)
    slider_y = widgets.Slider(ax_y, 'Y', -4.0, 4.0, valinit=initial_y)
    slider_psi = widgets.Slider(ax_psi, 'Psi (deg)', -180, 180, valinit=initial_psi)

    def update(val):
        # Get current slider values
        loa = slider_loa.val
        beam = slider_beam.val
        x = slider_x.val
        y = slider_y.val
        psi = slider_psi.val
        
        # Create new ship shape and transform it
        ship_shape = SHIP(loa, beam)
        transformed_shape = get_shape_at_xypsi(x, y, psi, ship_shape, degrees=True)
        
        # Update plot
        line.set_data(*zip(*transformed_shape))
        fig.canvas.draw()

    # Connect sliders to update function
    slider_loa.on_changed(update)
    slider_beam.on_changed(update)
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_psi.on_changed(update)

    plt.show()