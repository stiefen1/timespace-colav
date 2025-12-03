import numpy as np
from math import pi

DEG2RAD = lambda angle: np.deg2rad(angle)
RAD2DEG = lambda angle: np.rad2deg(angle)

def rotation_matrix(psi: float, degrees: bool = False) -> np.ndarray:
    """
    R = R(psi) computes the Euler angle rotation matrix R in SO(3)
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
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + pi) % (2 * pi) - pi
        
    return angle

def wrap_min_max(x: float | np.ndarray, x_min: float | np.ndarray, x_max: float | np.ndarray) -> float | np.ndarray:
    """Wraps input x to [x_min, x_max)

    Args:
        x (float or np.ndarray): Unwrapped value
        x_min (float or np.ndarray): Minimum value
        x_max (float or np.ndarray): Maximum value

    Returns:
        float or np.ndarray: Wrapped value
    """
    if isinstance(x, np.ndarray):
        return x_min + np.mod(x - x_min, x_max - x_min)
    else:
        return x_min + (x - x_min) % (x_max - x_min)

def wrap_angle_to_pmpi(angle: float | np.ndarray, degrees: bool = False) -> float | np.ndarray:
    """Wraps input angle to [-pi, pi) or [-180, 180) if degrees=True

    Args:
        angle (float or np.ndarray): Angle in radians

    Returns:
        float or np.ndarray: Wrapped angle
    """
    angle = DEG2RAD(angle) if degrees else angle
    if isinstance(angle, np.ndarray):
        out = wrap_min_max(angle, -np.pi * np.ones(angle.size), np.pi * np.ones(angle.size))
    else:
        out = wrap_min_max(angle, -np.pi, np.pi)
    return RAD2DEG(out) if degrees else out