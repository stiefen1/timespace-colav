from colav.obstacles import MovingShip
from colav.planner import TimeSpaceColav
import matplotlib.pyplot as plt, logging, colav, numpy as np
from itertools import combinations
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
    u=6,                    # Surge speed [m/s]
    v=0,                    # Sway speed [m/s]
    loa=40,                 # Length overall [m]
    beam=16,                # [m]
    degrees=False,          # Whether psi and cog are provided in degrees (True) or radians (False)
    mmsi=265041000          # Maritime Mobile Service Identity (Aurora AF Helsingborg ferry)
)

# Construct fake obstacles as shore: use .buffer to add safety margin.
shore = [
    Polygon([(0, -10), (30, -10), (30, 20), (0, 20), (0, -10)]),  # Square obstacle
    Point(-50, -50).buffer(20)                              # Circle obstacle
]

safety_distance = 10 # Minimal distance w.r.t ship [m] -> very small here, should be at least > length overall

ts1_with_sd = ts1.buffer(safety_distance, minkowski=True).simplify(0.5) # .simplify(2)   # Add safety margin using Minkowski sum
ts2_with_sd = ts2.buffer(safety_distance, minkowski=True).simplify(0.5)   # Add safety margin using Minkowski sum
shore_with_sd = [Polygon(obs.buffer(safety_distance).simplify(1).boundary.coords) for obs in shore] # Add safety margin to the shore using Minkowski sum

# Try swapping x values to see the result
p0 = (100, -40) # Own ship position
pf = (-100, 30) # Target position

planner = TimeSpaceColav(
    desired_speed=3,            # Desired speed
    distance_threshold=1000,    # Minimal distance to include target ships in trajectory planning
    shore=shore_with_sd,        # All the static obstacles with safety margin
    max_speed=5,                # Maximum speed
    max_course_rate=1,          # Max course rate
    max_iter=10,                # Max number of iterations
    colregs=True                # Whether to account for COLREGs or not
)

traj, info = planner.get(
    p0=p0,                                  # Inital position of own ship
    pf=pf,                                  # Target position of own ship 
    obstacles=[ts1_with_sd, ts2_with_sd],   # Moving obstacles
    heading=-70,
    margin=0
)  

def _circle_from_3_points(p1, p2, p3):
    """Circumscribed circle through 3 non-collinear points; returns ((cx, cy), r) or None."""
    ax, ay = p1;  bx, by = p2;  cx, cy = p3
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-10:
        return None  # collinear
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D
    return (ux, uy), float(np.hypot(ax - ux, ay - uy))


def _fit_circle_fixed_radius(critical_points, full_obstacle, radius):
    """
    Find the centre of a circle with the given fixed radius such that:
      - as many critical points as possible lie ON the circle boundary
      - all critical points are CONTAINED within the circle
    When fewer than 3 critical points exist, the closest obstacle vertices are
    used to help orient the centre.  Returns ((cx, cy), radius) or None.
    """
    R  = radius
    n  = len(critical_points)
    if n == 0:
        return None

    cps_arr = [np.array(p) for p in critical_points]

    # Feasibility: no two critical points can be more than 2R apart.
    for i, j in combinations(range(n), 2):
        if np.linalg.norm(cps_arr[i] - cps_arr[j]) > 2 * R + 1e-10:
            return None

    # Build candidate set: critical points + closest obstacle vertices (when n < 3)
    candidates = list(critical_points)
    if n < 3:
        verts = np.array(full_obstacle)
        dists = np.min(np.linalg.norm(verts[:, None, :] - np.array(critical_points)[None, :, :], axis=2), axis=1)
        added = 0
        for idx in np.argsort(dists):
            if dists[idx] > 1e-6:
                candidates.append(tuple(full_obstacle[idx]))
                added += 1
            if added == 3 - n:
                break
    cands_arr = [np.array(p) for p in candidates]

    best_center = None
    best_score  = -1
    tol         = 1e-6 * R + 1e-8  # tolerance for "on boundary"

    # Try every pair in the candidate set as boundary points.
    # Each pair defines <=2 candidate centres at distance R from both points.
    for i, j in combinations(range(len(candidates)), 2):
        p1, p2 = cands_arr[i], cands_arr[j]
        d = np.linalg.norm(p2 - p1)
        if d > 2 * R + 1e-10 or d < 1e-10:
            continue
        mid  = (p1 + p2) / 2
        h    = np.sqrt(max(R**2 - (d / 2)**2, 0.0))
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d
        for sign in [1.0, -1.0]:
            center   = mid + sign * h * perp
            cp_dists = np.array([np.linalg.norm(q - center) for q in cps_arr])
            if np.all(cp_dists <= R + 1e-10):           # feasible
                on_bnd = int(np.sum(np.abs(cp_dists - R) <= tol))
                if on_bnd > best_score:
                    best_score  = on_bnd
                    best_center = tuple(center)

    # Fallback when n == 1 and no pair gave a feasible centre:
    # place the centre at distance R toward the obstacle centroid.
    if best_center is None and n == 1:
        cp        = cps_arr[0]
        centroid  = np.mean(np.array(full_obstacle), axis=0)
        direction = centroid - cp
        norm      = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-10 else np.array([1.0, 0.0])
        best_center = tuple(cp + R * direction)

    return (best_center, R) if best_center is not None else None


def _fit_circle_min_radius(critical_points, full_obstacle, min_radius):
    """
    Find the best-fit circle with radius >= min_radius.
    The effective radius is max(natural_radius, min_radius), where natural_radius
    is the smallest circumscribed circle through 3 candidate points that contains
    all critical points.  The centre is then chosen to maximise the number of
    critical points that lie on the boundary.
    """
    n = len(critical_points)
    if n == 0:
        return None

    # Build the same candidate set as _fit_circle_fixed_radius
    candidates = list(critical_points)
    if n < 3:
        verts = np.array(full_obstacle)
        dists = np.min(
            np.linalg.norm(verts[:, None, :] - np.array(critical_points)[None, :, :], axis=2), axis=1
        )
        added = 0
        for idx in np.argsort(dists):
            if dists[idx] > 1e-6:
                candidates.append(tuple(full_obstacle[idx]))
                added += 1
            if added == 3 - n:
                break

    cps_arr = [np.array(p) for p in critical_points]

    # Smallest circumscribed circle through 3 candidates that contains all critical points
    nat_r = None
    for triple in combinations(candidates, 3):
        result = _circle_from_3_points(*triple)
        if result is None:
            continue
        center, r = result
        if all(np.linalg.norm(q - np.array(center)) <= r + 1e-10 for q in cps_arr):
            if nat_r is None or r < nat_r:
                nat_r = r

    R_eff = max(nat_r if nat_r is not None else min_radius, min_radius)
    return _fit_circle_fixed_radius(critical_points, full_obstacle, R_eff)


obs = {}
for node in planner.path_planner.path_nodes: # type: ignore
    if node['id'] == 0 or node['id'] == -1:
        continue
    elif node['id'] in obs.keys():
        obs[node['id']]['critical_points'].append(tuple(node['pos']))
    else:
        obs[node['id']] = {}
        obs[node['id']]['critical_points'] = [tuple(node['pos'])]
        obs[node['id']]['full_obstacle'] = list(zip(*planner.path_planner.obstacles[node['id']].exterior.coords.xy))
    # print(node.keys(), node.values())

print(list(obs.keys()), list(obs.values()))    

min_circle_radius = 20  # minimum radius; grows automatically if geometry requires it

circles = {}
for obs_id, data in obs.items():
    result = _fit_circle_min_radius(data['critical_points'], data['full_obstacle'],
                                    min_radius=min_circle_radius)
    if result is not None:
        circles[obs_id] = result   # ((cx, cy), radius)

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

from matplotlib.patches import Circle as MplCircle
for i, (obs_id, (center, radius)) in enumerate(circles.items()):
    ax.add_patch(MplCircle(center, radius, fill=False, edgecolor='magenta',
                           linewidth=1.5, linestyle='--',
                           label='fitted circle' if i == 0 else None))
    ax.plot(*center, 'x', color='magenta', markersize=8)

ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_title(f"Target ships, timespace footprints and static obstacles with margin")
ax.set_aspect('equal')
ax.legend()
plt.show()