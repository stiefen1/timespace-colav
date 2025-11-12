import numpy as np

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