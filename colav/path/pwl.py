from typing import List, Tuple, Optional
from shapely import LineString
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
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
