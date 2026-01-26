from colav.timespace import Plane
from colav.obstacles import MovingObstacle, MovingShip
from colav.planner import TimeSpaceColav
import matplotlib.pyplot as plt, logging, colav, numpy as np
from shapely import Polygon, Point, affinity
colav.configure_logging(level=logging.INFO)

# From Course & Speed Over Ground (CSOG)
ts = MovingShip.from_csog(
    position=(15, 20),    # (x, y) [m]
    psi=-125,                # Heading
    cog=-130,                # Course over ground
    sog=2.8,                  # Speed over ground [m/s]
    loa=20,                 # length overall [m]
    beam=6,                 # [m]
    degrees=True,           # Whether psi and cog are provided in degrees (True) or radians (False)
    mmsi=None,               # Maritime Mobile Service Identity
    du=0.1,
    dchi=5
)

p0 = (-20, -90) # Own ship position
pf = (-50, 100) # Target position

os = MovingShip.from_csog(
    position=p0,
    psi=-20,
    cog=-20,
    sog=3,
    loa=20,
    beam=6,
    degrees=True,
)


# Construct fake obstacles as shore: use .buffer to add safety margin.
shore = [
    affinity.translate(affinity.rotate(Polygon([(0, 0), (10, 0), (10, 20), (0, 10), (0, 0)]), -50), -60, 20), # Square obstacle
]

safety_distance = 10 # Minimal distance w.r.t ship [m] -> very small here, should be at least > length overall

ts_with_sd = ts.buffer(safety_distance) 
shore_with_sd = [obs.buffer(safety_distance, join_style='mitre') for obs in shore] # Add safety margin to the shore using Minkowski sum


planner = TimeSpaceColav(
    desired_speed=os.u,            # Desired speed
    distance_threshold=1000,    # Minimal distance to include target ships in trajectory planning
    shore=shore_with_sd,        # All the static obstacles with safety margin
    max_speed=3.5,
    max_yaw_rate=0.5,
    max_iter=50,
    colregs=True,
)

traj, info = planner.get(
    p0=p0,                                  # Inital position of own ship
    pf=pf,                                # Target position of own ship 
    # desired_heading=-75,
    obstacles=[ts_with_sd],   # Moving obstacles
    heading=os.psi,
)  

# Display target ships with their projected footprint
_, ax = plt.subplots(figsize=(7, 7))

# Start and target position (Own ship)
# ax.scatter(*p0, c='green', label='p0 (own ship)')
# ax.scatter(*info['pf'], c='purple', label='pf (own ship)', s=100)

# Target ships
ts.fill(ax=ax, c='orange', label='TS')
ts_with_sd.plot('--', ax=ax, c='orange')

if planner.path_planner is not None:
    planner.path_planner.plot(ax=ax, node_size=20, node_color='black')

# Projected footprint
for i, projected_ship in enumerate(planner.projector.get(p0, info['pf'], [ts_with_sd])):
    ax.fill(*projected_ship.exterior.xy, c='orange', alpha=0.3, label=f"footprints" if i==0 else None)

for j, obs in enumerate(shore):
    ax.fill(*obs.exterior.xy, c='forestgreen', label="obstacles" if j==0 else None)
    ax.plot(*shore_with_sd[j].exterior.xy, '--', c='forestgreen')

if traj is not None:
    traj.plot(ax=ax, c='red', label="trajectory")

# os.plot(ax=ax, c='blue')
os.fill(ax=ax, c='#4472C4', label="OS", alpha=1)

ax.set_xlim((-95, 45))
ax.set_ylim((-105, 105))
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
# ax.set_title(f"Target ships, time-space footprints and static obstacles with margin")
ax.set_aspect('equal')
# ax.legend()
plt.show()