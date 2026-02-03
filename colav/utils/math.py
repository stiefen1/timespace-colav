"""
Mathematical utilities for maritime navigation.

Provides common mathematical functions for angle handling, coordinate
transformations, and geometric calculations in maritime applications.

Key Functions
-------------
rotation_matrix : 2D rotation matrix generation
ssa : Smallest signed angle normalization
wrap_min_max : Value wrapping to specified range
wrap_angle_to_pmpi : Angle wrapping to [-π, π) or [-180°, 180°)
DEG2RAD, RAD2DEG : Angle unit conversion
"""

import numpy as np
from math import pi

DEG2RAD = lambda angle: np.deg2rad(angle)
RAD2DEG = lambda angle: np.rad2deg(angle)

def rotation_matrix(psi: float, degrees: bool = False) -> np.ndarray:
    """
    Generate 2D rotation matrix for given angle.
    
    Parameters
    ----------
    psi : float
        Rotation angle in radians (or degrees if degrees=True).
    degrees : bool, default False
        Whether input angle is in degrees.
        
    Returns
    -------
    ndarray
        3x3 rotation matrix in SO(3) for 2D rotations.
        
    Examples
    --------
    >>> R = rotation_matrix(90, degrees=True)
    >>> R = rotation_matrix(np.pi/2)
    """
    psi = np.deg2rad(psi) if degrees else psi
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    return np.array([
        [ c_psi, -s_psi, 0],
        [ s_psi, c_psi, 0],
        [ 0, 0, 1]
    ])

def ssa(angle: float) -> float:
    """
    Normalize angle to smallest signed angle in [-π, π).
    
    Parameters
    ----------
    angle : float
        Input angle in radians.
        
    Returns
    -------
    float
        Equivalent angle in [-π, π) range.
        
    Examples
    --------
    >>> ssa(3*np.pi)  # Returns -π
    >>> ssa(np.pi/2)  # Returns π/2
    """
    angle = (angle + pi) % (2 * pi) - pi
        
    return angle

def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap values to specified range [x_min, x_max).
    
    Parameters
    ----------
    x : float or ndarray
        Value(s) to wrap.
    x_min : float or ndarray
        Minimum value of target range.
    x_max : float or ndarray
        Maximum value of target range (exclusive).
        
    Returns
    -------
    float or ndarray
        Wrapped value(s) in [x_min, x_max) range.
        
    Examples
    --------
    >>> wrap_min_max(370, 0, 360)  # Returns 10
    >>> wrap_min_max(-10, 0, 360)  # Returns 350
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)

def wrap_angle_to_pmpi(angle: float | np.ndarray, degrees: bool = False) -> float | np.ndarray:
    """
    Wrap angles to [-π, π) or [-180°, 180°) range.
    
    Parameters
    ----------
    angle : float or ndarray
        Input angle(s) in radians (or degrees if degrees=True).
    degrees : bool, default False
        Whether input/output is in degrees.
        
    Returns
    -------
    float or ndarray
        Wrapped angle(s) in appropriate range.
        
    Examples
    --------
    >>> wrap_angle_to_pmpi(3*np.pi)  # Returns -π
    >>> wrap_angle_to_pmpi(270, degrees=True)  # Returns -90
    """
    angle = DEG2RAD(angle) if degrees else angle
    if isinstance(angle, np.ndarray):
        out = wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        out = wrap_min_max(angle, -np.pi, np.pi)
    return RAD2DEG(out) if degrees else out