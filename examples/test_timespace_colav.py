from colav.timespace import Plane
from colav.obstacles import MovingObstacle, MovingShip
from colav.planner import TimeSpaceColav
import matplotlib.pyplot as plt, logging, colav, numpy as np
from shapely import Polygon, Point
colav.configure_logging(level=logging.INFO)

# From Course & Speed Over Ground (CSOG)
ts1 = MovingShip.from_csog(
    position=(-60, 120),    # (x, y) [m]
    psi=170,                # Heading
    cog=165,                # Course over ground
    sog=3,                  # Speed over ground [m/s]
    loa=20,                 # length overall [m]
    beam=6,                 # [m]
    degrees=True,           # Whether psi and cog are provided in degrees (True) or radians (False)
    mmsi=None               # Maritime Mobile Service Identity
)

# From surge & sway speed
ts2 = MovingShip.from_body(
    position=(20, -100),    # (x, y) [m]
    psi=np.pi/8,            # Heading
    u=8,                    # Surge speed [m/s]
    v=0,                    # Sway speed [m/s]
    loa=40,                 # Length overall [m]
    beam=16,                # [m]
    degrees=False,          # Whether psi and cog are provided in degrees (True) or radians (False)
    mmsi=265041000          # Maritime Mobile Service Identity (Aurora AF Helsingborg ferry)
)

# Construct fake obstacles as shore: use .buffer to add safety margin.
shore = [
    Polygon([(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]), # Square obstacle
    Point(-50, -50).buffer(20) # Circle obstacle
]

safety_distance = 10 # Minimal distance w.r.t ship [m] -> very small here, should be at least > length overall

ts1_with_sd = ts1.buffer(safety_distance).simplify(2)   # Add safety margin using Minkowski sum
ts2_with_sd = ts2.buffer(safety_distance).simplify(2)   # Add safety margin using Minkowski sum
shore_with_sd = [obs.buffer(safety_distance).simplify(2) for obs in shore] # Add safety margin to the shore using Minkowski sum

# Try swapping x values to see the result
p0 = (100, -40) # Own ship position
pf = (-100, 30) # Target position

planner = TimeSpaceColav(
    desired_speed=3,            # Desired speed
    distance_threshold=1000,    # Minimal distance to include target ships in trajectory planning
    shore=shore_with_sd,        # All the static obstacles with safety margin
    max_speed=5,
    max_yaw_rate=1,
    max_iter=10,
    colregs=True
)

traj, info = planner.get(
    p0=p0,                                  # Inital position of own ship
    pf=pf,                                # Target position of own ship 
    desired_heading=-75,
    obstacles=[ts1_with_sd, ts2_with_sd],   # Moving obstacles
    heading=-70
)  

# Display target ships with their projected footprint
_, ax = plt.subplots(figsize=(7, 7))

# Start and target position (Own ship)
ax.scatter(*p0, c='green', label='p0 (own ship)')
ax.scatter(*info['pf'], c='purple', label='pf (own ship)')

# Target ships
ts1.fill(ax=ax, c='blue', label='target ship 1')
ts2.fill(ax=ax, c='red', label='target ship 2')
ts1_with_sd.fill(ax=ax, c='blue', alpha=0.5)
ts2_with_sd.fill(ax=ax, c='red', alpha=0.5)

if planner.path_planner is not None:
    planner.path_planner.plot(ax=ax, node_size=20)

# Projected footprint
for i, projected_ship in enumerate(planner.projector.get(p0, info['pf'], [ts1_with_sd, ts2_with_sd])):
    ax.fill(*projected_ship.exterior.xy, c='grey', alpha=0.7, label=f"footprints" if i==0 else None)

for j, obs in enumerate(shore):
    ax.fill(*obs.exterior.xy, c='orange', label="obstacles" if j==0 else None)
    ax.fill(*shore_with_sd[j].exterior.xy, c='orange', alpha=0.5)

if traj is not None:
    traj.plot(ax=ax, c='red', label="trajectory")

# ax.set_xlim((-150, 150))
# ax.set_ylim((-150, 150))
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title(f"Target ships, time-space footprints and static obstacles with margin")
ax.set_aspect('equal')
ax.legend()
plt.show()