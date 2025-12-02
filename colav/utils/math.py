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