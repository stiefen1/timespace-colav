from typing import List, Tuple, Optional
from shapely import LineString
from matplotlib.axes import Axes
import matplotlib.pyplot as plt, numpy as np
"""
Typical use case:

"""

class PWLTrajectory:
    def __init__(
            self,
            xyt: List[ Tuple[float, float, float]]
    ):
        self._linestring = LineString(xyt)

    def interpolate(self, distance: float, normalized=False) -> Tuple[float, float, float]:
        point = self._linestring.interpolate(distance, normalized=normalized)
        return point.x, point.y, point.z
    
    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    def scatter(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    def get_heading(self, distance: float, normalized: bool = False, degrees:bool = False) -> float:
        """
        Returns the orientation of the backbone at a given progression value. Useful for integrating power consumption along corridor.
        """
        length = 1 if normalized else self._linestring.length
        assert 0 <= distance <= length , f"distance must be within [0, {length:.1f}]. Got distance={distance:.2f}"
        
        if distance == 0:
            prog_prev = 0
            prog_next = 0.005 * length
        else: # meaning 0 < progression <= length
            prog_prev = 0.995 * distance
            prog_next = distance

        p1 = np.array(self._linestring.interpolate(prog_prev, normalized=normalized).xy).squeeze()
        p2 = np.array(self._linestring.interpolate(prog_next, normalized=normalized).xy).squeeze()

        heading_rad = np.atan2(p2[0]-p1[0], p2[1]-p1[1])
        return np.rad2deg(heading_rad) if degrees else heading_rad
    
    def get_speed(self, distance: float, normalized: bool = False) -> float:
        """
        """
        length = 1 if normalized else self._linestring.length
        assert 0 <= distance <= length , f"distance must be within [0, {length:.1f}]. Got distance={distance:.2f}"
        
        if distance == 0:
            prog_prev = 0
            prog_next = 0.005 * length
        else: # meaning 0 < progression <= length
            prog_prev = 0.995 * distance
            prog_next = distance

        p1 = self._linestring.interpolate(prog_prev, normalized=normalized)
        p2 = self._linestring.interpolate(prog_next, normalized=normalized)
        return self.get_edge_speed((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))
    
    def get_edge_speed(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 / (p2[2] - p1[2])

    @property
    def xyt(self) -> List[ Tuple[float, float, float] ]:
        return [(point[0], point[1], point[2]) for point in self._linestring.coords]
    
    
class PWLPath:
    def __init__(
            self,
            xy: List[ Tuple[float, float] ]
    ):
        self._linestring = LineString(xy)

    def interpolate(self, distance: float, normalized=False) -> Tuple[float, float]:
        point = self._linestring.interpolate(distance, normalized=normalized)
        return point.x, point.y
    
    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    def scatter(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(*self._linestring.coords.xy, *args, **kwargs)
        return ax
    
    @property
    def xy(self) -> List[ Tuple[float, float] ]:
        return [(point[0], point[1]) for point in self._linestring.coords]

if __name__ == "__main__":
    from colav.timespace.projector import TimeSpaceProjector
    wpts = [
        (0, 0, 0),
        (2, 1, 1),
        (2.5, 0.5, 1.5),
        (3, 0, 2.5)
    ]

    path = PWLPath([(wpt[0], wpt[1]) for wpt in wpts])
    traj = PWLTrajectory(wpts)
    proj = TimeSpaceProjector(3)
    print(proj.get((0, 0), (2, 0), []))
    print(proj.add_timestamps(path).xyt)
    print(path.interpolate(0.5, normalized=True), traj._linestring.interpolate(0.5, normalized=True))
    print(path.xy, traj.xyt)
