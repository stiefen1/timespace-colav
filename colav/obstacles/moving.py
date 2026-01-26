from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
from colav.obstacles.transform import get_shape_at_xypsi
from colav.obstacles.shapes import SHIP
from colav.utils.math import rotation_matrix, DEG2RAD, RAD2DEG
from colav.utils import generate_random_mmsi
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely, logging
from shapely.affinity import rotate
from math import cos, sin
from copy import deepcopy
logger = logging.getLogger(__name__)

# TODO: Implement Moving Ship with uncertainties
class MovingObstacle:
    def __init__(
        self,
        position: Tuple[float, float],
        psi: float,
        velocity: Tuple[float, float],
        geometry_at_psi_equal_0: List[ Tuple[float, float] ],
        degrees: bool = False,
        mmsi: Optional[int] = None
    ):
        self.position = position
        self.psi = psi
        self.velocity = velocity
        self.geometry_at_psi_equal_0 = geometry_at_psi_equal_0
        self.degrees = degrees
        self.mmsi: int = mmsi or -generate_random_mmsi() # Negative mmsi to highlight fake ships
        
        self.reset_geometry()
        
        # Compute speed in body frame
        R = rotation_matrix(self.psi, degrees=self.degrees)
        uv = R.T[0:2, 0:2] @ np.array([self.velocity[1], self.velocity[0]])
        self.u = float(uv[0])
        self.v = float(uv[1])

    def reset_geometry(self) -> None:
         # Get the transformed geometry points 
        self.geometry = get_shape_at_xypsi(*self.position, self.psi, self.geometry_at_psi_equal_0, degrees=self.degrees)

    def distance(self, x: float, y: float) -> float:
        """Returns distance of TS from (x, y)"""
        return ((x-self.position[0])**2 + (y-self.position[1])**2)**0.5
    
    def predict(self, dt: float, model: Literal['CVM'] = 'CVM', **kwargs) -> "MovingObstacle":
        match model:
            case 'CVM':
                return MovingObstacle(
                    (
                        self.position[0] + self.velocity[0] * dt,
                        self.position[1] + self.velocity[1] * dt
                    ),
                    self.psi,
                    self.velocity,
                    self.geometry_at_psi_equal_0,
                    degrees=self.degrees,
                    mmsi=self.mmsi
                )
            case _:
                logger.warning(f"{model} is not a valid prediction model option.")
                return None

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
    
    def simplify(self, tolerance: float, preserve_topology: bool = True, **kwargs) -> "MovingObstacle":
        """
        """
        new_self = deepcopy(self)
        new_geometry_as_polygon = shapely.simplify(shapely.Polygon(self.geometry_at_psi_equal_0), tolerance, preserve_topology=preserve_topology, **kwargs)
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
    
    @property
    def vertices_velocity(self) -> List[ Tuple[float, float] ]:
        return len(self.geometry) * [self.velocity]
    
    @property
    def robust_geometry(self) -> Optional[List[ Tuple[float, float] ]]:
        return None
    
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
        mmsi: int | None = None,
        dchi: Optional[float] = None,
        du: Optional[float] = None 
    ):
        
        if dchi is not None:
            assert dchi >= 0, f"dchi must be greater or equal to 0. Got dchi={dchi}"
        if du is not None:
            assert du >= 0, f"du must be greater or equal to 0. Got du={du}"

        self.loa = loa
        self.beam = beam
        self.dchi = dchi
        self.du = du
        super().__init__(position, psi, velocity, SHIP(loa, beam), degrees=degrees, mmsi=mmsi)

    @staticmethod
    def from_body(position: Tuple[float, float], psi: float, u: float, v: float, loa: float, beam: float, *args, degrees: bool = False, mmsi: Optional[int] = None, dchi: Optional[float] = None, du: Optional[float] = None , **kwargs) -> "MovingShip":
        """
        Return a MovingShip object using speed in body frame.
        """
        
        R = rotation_matrix(psi, degrees=degrees)
        velocity = R[0:2, 0:2] @ np.array([u, v]) # velocity in N-E frame
        return MovingShip(position, psi, (float(velocity[1]), float(velocity[0])), loa, beam, degrees=degrees, mmsi=mmsi, du=du, dchi=dchi)
    
    @staticmethod
    def from_csog(position: Tuple[float, float], psi: float, cog: float, sog: float, loa: float, beam: float, *args, degrees: bool = False, mmsi: Optional[int] = None, dchi: Optional[float] = None, du: Optional[float] = None, **kwargs) -> "MovingShip":
        # Convert course-over-ground, speed-over-ground into u, v
        cog = np.deg2rad(cog) if degrees else cog
        x_dot = sog * np.sin(cog) # = east speed
        y_dot = sog * np.cos(cog) # = north speed
        return MovingShip(position, psi, (x_dot, y_dot), loa=loa, beam=beam, degrees=degrees, mmsi=mmsi, du=du, dchi=dchi)
    
    def predict(self, dt: float, model: Literal['CVM'] = 'CVM', **kwargs) -> "MovingShip":
        match model:
            case 'CVM':
                return MovingShip(
                    (
                        self.position[0] + self.velocity[0] * dt,
                        self.position[1] + self.velocity[1] * dt
                    ),
                    self.psi,
                    self.velocity,
                    self.loa,
                    self.beam,
                    degrees=self.degrees,
                    mmsi=self.mmsi,
                    du=self.du,
                    dchi=self.dchi
                )
            case _:
                logger.warning(f"{model} is not a valid prediction model option.")
                return None
            
    def resample_velocity(self) -> "MovingShip":
        if self.du is None or self.dchi is None:
            logger.warning(f"resample_velocity has no effect if du or dchi are not specified.")
            return self
        
        du = np.random.normal(loc=0, scale=self.du / 3) # Divide by 3 such that ~99.7% remain within +-self.du
        dchi = np.random.normal(loc=0, scale=self.dchi / 3) # Divide by 3 such that ~99.7% remain within +-self.dchi

        return MovingShip.from_body(self.position, self.psi + dchi, self.u + du, self.v, self.loa, self.beam, degrees=self.degrees, mmsi=self.mmsi, dchi=self.dchi, du=self.du)
            
    def buffer(self, distance: float, minkowski: bool = False, **kwargs) -> "MovingShip":
        if minkowski:
            new_ship = deepcopy(self)
            new_ship.geometry_at_psi_equal_0 = [(x, y) for x, y in shapely.Polygon(new_ship.geometry_at_psi_equal_0).buffer(distance, **kwargs).exterior.coords]
            new_ship.reset_geometry()
            return new_ship

        return MovingShip.from_body(self.position, self.psi, self.u, self.v, self.loa + 2 * distance, self.beam + 2 * distance, degrees=self.degrees, mmsi=self.mmsi, dchi=self.dchi, du=self.du)

    def get_robust_geometry(self) -> Optional[List[ Tuple[float, float] ]]:
        vel_norm = np.linalg.norm(np.array(self.velocity))
        if vel_norm == 0:
            logger.warning(f"Impossible to compute robust geometry because ship speed is {vel_norm} <= 0")
            return self.geometry
        
        robust_geometry = None
        if self.dchi is not None:
            poly_left = rotate(shapely.Polygon(self.geometry), self.dchi / 2, origin=self.position, use_radians=not(self.degrees))
            poly_left_left = rotate(shapely.Polygon(self.geometry), self.dchi, origin=self.position, use_radians=not(self.degrees))
            poly_center = self.geometry
            poly_right = rotate(shapely.Polygon(self.geometry), -self.dchi / 2, origin=self.position, use_radians=not(self.degrees))
            poly_right_right = rotate(shapely.Polygon(self.geometry), -self.dchi, origin=self.position, use_radians=not(self.degrees))

            vertex_0 = poly_center[0]
            vertex_1 = poly_right.exterior.coords[0]
            vertex_2 = poly_right_right.exterior.coords[0]
            vertex_3 = poly_right_right.exterior.coords[1]
            vertex_4 = poly_right_right.exterior.coords[2]
            vertex_5 = poly_right.exterior.coords[3]
            vertex_6 = poly_center[3]
            vertex_7 = poly_left.exterior.coords[3]
            vertex_8 = poly_left_left.exterior.coords[4]
            vertex_9 = poly_left_left.exterior.coords[5]
            vertex_10 = poly_left_left.exterior.coords[6]
            vertex_11 = poly_left.exterior.coords[0]

            robust_geometry = [
                vertex_0,
                vertex_1,
                vertex_2,
                vertex_3,
                vertex_4,
                vertex_5,
                vertex_6,
                vertex_7,
                vertex_8,
                vertex_9,
                vertex_10,
                vertex_11,
                vertex_0
            ]

        return robust_geometry

    def get_robust_vertices_velocity(self) -> List[ Tuple[float, float] ]:
        assert self.du is not None and self.dchi is not None, f"du and dchi must be specified. Got du={self.du} and dchi={self.dchi}."
        u_max = self.u + self.du
        u_min = max(self.u - self.du, 0)

        vel_center = np.array(self.velocity)
        vel_norm = np.linalg.norm(vel_center)
        if vel_norm == 0:
            logger.warning(f"Impossible to compute robust geometry because ship speed is {vel_norm} <= 0")
            return super().vertices_velocity

        normalized_vel_center = vel_center / vel_norm
        normalized_vel_left = rotation_matrix(self.dchi / 2, degrees=self.degrees)[0:2, 0:2] @ normalized_vel_center
        normalized_vel_left_left = rotation_matrix(self.dchi, degrees=self.degrees)[0:2, 0:2] @ normalized_vel_center
        normalized_vel_right = rotation_matrix(-self.dchi / 2, degrees=self.degrees)[0:2, 0:2] @ normalized_vel_center
        normalized_vel_right_right = rotation_matrix(-self.dchi, degrees=self.degrees)[0:2, 0:2] @ normalized_vel_center

        vel_0 = tuple((normalized_vel_center * u_max).tolist())
        vel_1 = tuple((normalized_vel_right * u_max).tolist())
        vel_2 = tuple((normalized_vel_right_right * u_max).tolist())
        vel_3 = tuple((normalized_vel_right_right * u_max).tolist())
        vel_4 = tuple((normalized_vel_right_right * u_min).tolist())
        vel_5 = tuple((normalized_vel_right * u_min).tolist())
        vel_6 = tuple((normalized_vel_center * u_min).tolist())
        vel_7 = tuple((normalized_vel_left * u_min).tolist())
        vel_8 = tuple((normalized_vel_left_left * u_min).tolist())
        vel_9 = tuple((normalized_vel_left_left * u_max).tolist())
        vel_10 = tuple((normalized_vel_left_left * u_max).tolist())
        vel_11 = tuple((normalized_vel_left * u_max).tolist())

        velocities = [
            vel_0,
            vel_1,
            vel_2,
            vel_3,
            vel_4,
            vel_5,
            vel_6,
            vel_7,
            vel_8,
            vel_9,
            vel_10,
            vel_11,
            vel_0
        ]

        return velocities


    @property
    def ne(self) -> Tuple[float, float]:
        return self.position[1], self.position[0]
    
    @property
    def vertices_velocity(self) -> List[ Tuple[float, float] ]:
        if self.du is None or self.dchi is None:
            return super().vertices_velocity
        else: # Robust velocities
            return self.get_robust_vertices_velocity()
        
    @property
    def robust_geometry(self) -> Optional[List[ Tuple[float, float] ]]:
        return self.get_robust_geometry()


if __name__ == "__main__":
    from colav.obstacles.shapes import SHIP
    import matplotlib.pyplot as plt

    # Recommended way for declaring a ship
    loa, beam = 5, 2
    ship = MovingShip(
        (-2, 3),
        135,
        (1, -1),
        loa,
        beam,
        degrees=True,
        dchi=5,
        du=0.2
    )

    # Unrecommended way for declaring a ship
    target_ship = MovingObstacle.from_body((15, 10), 45, 1, 0, SHIP(loa, beam), degrees=True)
    print(target_ship.velocity, ship.u, ship.v)

    t0 = 10
    t1 = 20
    fig = plt.figure()
    ax = fig.add_subplot() if t0 is None else fig.add_subplot(projection='3d') 
    ship.plot(ax=ax, t=t0, c='purple')
    ship.fill(ax=ax, t=t0, facecolor='grey')
    ship.scatter(ax=ax, t=t0, c='purple')

    target_ship.plot(ax=ax, t=t0, c='green')
    target_ship.fill(ax=ax, t=t0, facecolor='grey')
    target_ship.scatter(ax=ax, t=t0, c='green')

    # Plot ships at t=t1
    ship.predict(t1-t0).plot(ax=ax, c='purple', t=t1)
    target_ship.predict(t1-t0).plot(ax=ax, c='green', t=t1)

    # Plot robust envelope
    if ship.robust_geometry is not None:
        ax.plot(*zip(*ship.robust_geometry))
        print(ship.get_robust_vertices_velocity())

    ax.set_aspect('equal')
    plt.show()