from typing import List, Tuple
import numpy as np, matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import logging
logger = logging.getLogger(__name__)

"""
Typical use case:

"""
class Plane:
    def __init__(
            self,
            p0: Tuple | List,
            pf: Tuple | List,
            t0: float,
            tf: float
    ):
        assert tf > t0, f"tf must be greater than t0. Got t0={t0}, tf={tf}"
        assert len(p0) == 2, f"p0 must be a tuple with len=2. Got p0={p0}"
        assert len(pf) == 2, f"pf must be a tuple with len=2. Got pf={pf}"
        assert (p0[0] != pf[0]) or (p0[1] != pf[1]), f"p0 and pf must have different values. Got p0={p0} and pf={pf}"
        self._p0_ts = (*p0, t0) # Initial position in timespace
        self._pf_ts = (*pf, tf) # Final position in timespace
        self._update_plane()
        logger.debug(f"Successfuly created Plane.")
        
    def _update_plane(self) -> None:
        """
        Calculate the vector parameters of the timespace plane: b_ts = [bx, by, bt].T
        
        """
        x0, y0, t0 = self._p0_ts
        xf, yf, tf = self._pf_ts

        # Solve linear system for bx, by, bt
        A = np.array([
            [x0, y0, 1, 0],
            [xf, yf, 1, 0],
            [1, 0, 0, x0-xf],
            [0, 1, 0, y0-yf]
        ])
        t_vec = np.array([t0, tf, 0, 0]).T

        if np.linalg.det(A) == 0:
            raise ValueError(f"det(A) = 0. Verify that p0 != pf")
        
        sol = np.linalg.inv(A) @ t_vec
        self._b_ts: np.ndarray = sol[0:3]
        self._alpha = float(sol[3])

        logger.debug(f"Plane was succesfully updated.")

    def get_time(self, x: float | int | np.ndarray, y: float | int | np.ndarray) -> float | np.ndarray:
        """
        Evaluate time at (x, y) according to the plane equation.
        """
        assert isinstance(x, float) or isinstance(x, int) or isinstance(x, np.ndarray), f"x must be a float, integer or a numpy array. Got type(x)={type(x)}"
        assert isinstance(y, float) or isinstance(y, int) or isinstance(y, np.ndarray), f"y must be a float, integer or a numpy array. Got type(x)={type(y)}"
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            assert x.shape == y.shape, f"x and y must have same shapes. Got x.shape={x.shape} and y.shape={y.shape}"

        bx, by, bt = self._b_ts.tolist()
        return bx*x + by*y + bt

    def intersection_with_single_vertex(self, vertex: Tuple[float, float], velocity: Tuple[float, float], robust: bool = False) -> Tuple[Tuple[float, float], float, bool]:
        """
        Compute the intersection between timespace plane and a single vertex moving at a given (constant) velocity.

        Check whether the spatio-temporal obstacle is in past (valid=False) or future (valid=True)
        """
        assert len(velocity) == 2, f"velocity must be a 2D tuple. Got velocity={velocity}"
        assert len(vertex) == 2, f"vertex must be a 2D tuple. Got vertex={vertex}"
        
        # Retrieve plane parameters
        bx, by, bt = self._b_ts
        b = np.array([bx, by])

        # Compute denominator, making sure we don't divide by 0
        p_dot = np.array(velocity)
        den = 1 - b @ p_dot
        if den == 0:
            raise ValueError(f"The obstacle is moving parallely to the plane. Try to slightly change its speed.")

        p_i = np.array(vertex)

        # Compute intersection coordinates according to equations (8.1) and (6.2)
        den = 1e-6 if (den < 0 and robust) else den
        t_i = ((b @ p_i) + bt) / den
        v_at_t_i = p_i + p_dot * t_i
        x_i, y_i = v_at_t_i[0], v_at_t_i[1]

        valid = False if t_i < 0 else True
                
        return (x_i, y_i), t_i, valid

    def intersection(self, vertices: List[ Tuple[float, float] ], velocities: List[ Tuple[float, float] ], robust: bool = False) -> Tuple[ List[ Tuple[float, float] ], List[float], bool ]:
        """
        Compute the intersection between timespace plane and a list of vertices moving at a given (constant) velocity.

        Check whether the spatio-temporal obstacle is in past (valid=False) or future (valid=True)
        """
        projected_vertices, times, all_valid = [], [], False
        for vertex, velocity in zip(vertices, velocities):
            p_i, t_i, valid = self.intersection_with_single_vertex(vertex, velocity, robust=robust)
            projected_vertices.append(p_i)
            times.append(t_i)
            all_valid = all_valid or valid
        return projected_vertices, times, all_valid
        
    def plot(self, *args, ax: Axes3D | None = None, xlim:Tuple[float, float] | None = None, ylim:Tuple[float, float] | None = None, zlim:Tuple[float, float] | None = None, grid:Tuple[int, int] = (10, 10), **kwargs) -> Axes:
        """
        Plot the timespace plane as a 3D surface.  
        """
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        x0, y0, t0 = self._p0_ts
        xf, yf, tf = self._pf_ts

        xlim = xlim or (min(x0, xf), max(x0, xf))
        ylim = ylim or (min(y0, yf), max(y0, yf))
        zlim = zlim or (min(t0, tf), max(t0, tf))

        # Evaluate time in xlim, ylim
        X = np.linspace(*xlim, grid[0]+1)
        Y = np.linspace(*ylim, grid[1]+1)
        XX, YY = np.meshgrid(X, Y)
        Z = self.get_time(XX, YY)

        ax.plot_surface(XX, YY, Z, *args, **kwargs)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        return ax
    
    @property
    def bx(self) -> float:
        return self._b_ts[0]
    
    @property
    def by(self) -> float:
        return self._b_ts[1]
    
    @property
    def bt(self) -> float:
        return self._b_ts[2]
    
    @property
    def b(self) -> Tuple[float, float, float]:
        return tuple(self._b_ts.tolist())
    
    @property
    def p0(self) -> Tuple[float, float]:
        return self._p0_ts[0], self._p0_ts[1]
    
    @property
    def pf(self) -> Tuple[float, float]:
        return self._pf_ts[0], self._pf_ts[1]
    
    @property
    def t0(self) -> float:
        return self._p0_ts[2]

    @property
    def tf(self) -> float:
        return self._pf_ts[2]
    
if __name__ == "__main__":
    t0, p0 = 1, (1, 2)
    tf, pf = 10, (4, 8)

    toy_obstacle = [(2., 5), (2, 6), (3, 6), (3, 5)]
    velocity = 1/10, 1/10
    
    plane = Plane(p0, pf, t0, tf)
    print(plane.intersection(toy_obstacle, len(toy_obstacle) * [velocity]))
    inter, ts, valid = plane.intersection(toy_obstacle, len(toy_obstacle) * [velocity])

    ax = plane.plot(alpha=0.3)
    ax.scatter(*zip(*toy_obstacle), c='green')
    ax.scatter(*zip(*inter), c='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    plt.show()