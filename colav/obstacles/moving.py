from dataclasses import dataclass
from typing import List, Tuple, Optional
from colav.obstacles.transform import get_shape_at_xypsi
from colav.obstacles.shapes import SHIP
from colav.utils.math import rotation_matrix
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely
from copy import deepcopy

class MovingObstacle:
    def __init__(
        self,
        position: Tuple[float, float],
        psi: float,
        velocity: Tuple[float, float],
        geometry_at_psi_equal_0: List[ Tuple[float, float] ],
        degrees: bool = False,
        mmsi: int | None = None
    ):
        self.position = position
        self.psi = psi
        self.velocity = velocity
        self.geometry_at_psi_equal_0 = geometry_at_psi_equal_0
        self.degrees = degrees
        self.mmsi = mmsi
        
        self.reset_geometry()
        
        # Compute speed in body frame
        R = rotation_matrix(self.psi, degrees=self.degrees)
        uv = R.T[0:2, 0:2] @ np.array([self.velocity[1], self.velocity[0]])
        self.u = float(uv[0])
        self.v = float(uv[1])

    def reset_geometry(self) -> None:
         # Get the transformed geometry points 
        # print(self.geometry_at_psi_equal_0) 
        self.geometry = get_shape_at_xypsi(*self.position, self.psi, self.geometry_at_psi_equal_0, degrees=self.degrees)

    def plot(self, *args, ax: Axes | Axes3D | None = None, t: float | None = 0, **kwargs) -> Axes | Axes3D:
        if ax is None:
            _, ax = plt.subplots()
        if isinstance(ax, Axes3D):
            ax.plot(*zip(*self.geometry), zs=t, *args, **kwargs)
        elif isinstance(ax, Axes):
            ax.plot(*zip(*self.geometry), *args, **kwargs)
        else:
            raise TypeError(f"ax must be an instance of Axes. Got type(ax)={type(ax)}")
        return ax
    
    def buffer(self, distance: float, **kwargs) -> "MovingObstacle":
        """
        join_style='mitre' is one of the keywords argument you can pass.
        """
        new_self = deepcopy(self)
        new_geometry_as_polygon = shapely.buffer(shapely.Polygon(self.geometry_at_psi_equal_0), distance, **kwargs)
        new_self.geometry_at_psi_equal_0 = list(zip(*new_geometry_as_polygon.exterior.xy))
        new_self.reset_geometry()
        return new_self
    
    @staticmethod
    def from_body(position: Tuple[float, float], psi: float, u: float, v: float, geometry_at_psi_equal_0: List[ Tuple[float, float] ], degrees: bool = False, mmsi: int | None = None) -> "MovingObstacle":
        """
        Return a MovingObstacle object using speed in body frame.
        """
        
        R = rotation_matrix(psi, degrees=degrees)
        velocity = R[0:2, 0:2] @ np.array([u, v]) # velocity in N-E frame
        return MovingObstacle(position, psi, (float(velocity[1]), float(velocity[0])), geometry_at_psi_equal_0, degrees=degrees, mmsi=mmsi)
    
    def fill(self, *args, ax: Axes | Axes3D | None = None, t: float | None = 0, **kwargs) -> Axes | Axes3D:
        if ax is None:
            _, ax = plt.subplots()
        if isinstance(ax, Axes3D):
            # build a list of 3D vertices (x, y, z=0) and pass it as a single polygon
            verts3d = [(x, y, t) for (x, y) in self.geometry]
            coll = Poly3DCollection([verts3d], *args, **kwargs)
            ax.add_collection3d(coll)
        elif isinstance(ax, Axes):
            ax.fill(*zip(*self.geometry), *args, **kwargs)
        else:
            raise TypeError(f"ax must be an instance of Axes. Got type(ax)={type(ax)}")
        return ax
    
    def scatter(self, *args, ax: Axes | Axes3D | None = None, t: float | None = 0, **kwargs) -> Axes | Axes3D:
        if ax is None:
            _, ax = plt.subplots()
        if isinstance(ax, Axes3D):
            ax.scatter(*zip(*self.geometry), zs=t, *args, **kwargs)
        elif isinstance(ax, Axes):
            ax.scatter(*zip(*self.geometry), *args, **kwargs)
        else:
            raise TypeError(f"ax must be an instance of Axes. Got type(ax)={type(ax)}")
        return ax
    
class MovingShip(MovingObstacle):
    """
    """
    def __init__(
        self,
        position: Tuple[float, float], # XY = East, North
        psi: float,
        velocity: Tuple[float, float], # World frame XY
        loa: float,
        beam: float,
        degrees: bool = False,
        mmsi: int | None = None
    ):
        super().__init__(position, psi, velocity, SHIP(loa, beam), degrees=degrees, mmsi=mmsi)

    @staticmethod
    def from_body(position: Tuple[float, float], psi: float, u: float, v: float, loa: float, beam: float, *args, degrees: bool = False, mmsi: int | None = None, **kwargs) -> "MovingObstacle":
        """
        Return a MovingObstacle object using speed in body frame.
        """
        
        R = rotation_matrix(psi, degrees=degrees)
        velocity = R[0:2, 0:2] @ np.array([u, v]) # velocity in N-E frame
        return MovingShip(position, psi, (float(velocity[1]), float(velocity[0])), loa, beam, degrees=degrees, mmsi=mmsi)
    
    @staticmethod
    def from_csog(position: Tuple[float, float], psi: float, cog: float, sog: float, loa: float, beam: float, *args, degrees: bool = False, mmsi: Optional[int] = None, **kwargs) -> "MovingObstacle":
        # Convert course-over-ground, speed-over-ground into u, v
        cog = np.deg2rad(cog)
        x_dot = sog * np.sin(cog) # = east speed
        y_dot = sog * np.cos(cog) # = north speed
        return MovingShip(position, psi, (x_dot, y_dot), loa=loa, beam=beam, degrees=degrees, mmsi=mmsi)


if __name__ == "__main__":
    from colav.obstacles import SHIP
    import matplotlib.pyplot as plt

    loa, beam = 5, 2
    ship = MovingShip(
        (-2, 3),
        135,
        (1, -1),
        loa,
        beam,
        degrees=True
    )

    target_ship = MovingObstacle.from_body((15, 10), 45, 1, 0, SHIP(loa, beam), degrees=True)
    print(target_ship.velocity, ship.u, ship.v)

    t = None
    fig = plt.figure()
    ax = fig.add_subplot() if t is None else fig.add_subplot(projection='3d') 
    ship.plot(ax=ax, t=t)
    ship.fill(ax=ax, t=t, facecolor='grey')
    ship.scatter(ax=ax, t=t, c='red')

    target_ship.plot(ax=ax, t=t)
    target_ship.fill(ax=ax, t=t, facecolor='red')
    target_ship.scatter(ax=ax, t=t, c='green')

    ax.set_aspect('equal')
    plt.show()