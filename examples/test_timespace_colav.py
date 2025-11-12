from colav.timespace import Plane
from colav.obstacles import MovingObstacle
from colav.planner import TimeSpaceColav
import matplotlib.pyplot as plt

# obs = MovingObstacle()
colav = TimeSpaceColav()


t0, p0 = 1, (1, 2)
tf, pf = 10, (4, 8)

toy_obstacle = [(2., 5), (2, 6), (3, 6), (3, 5)]
velocity = 1/10, 1/10

plane = Plane(p0, pf, t0, tf)
print(plane.intersection(toy_obstacle, velocity))
inter, ts = plane.intersection(toy_obstacle, velocity)

ax = plane.plot(alpha=0.3)
ax.scatter(*zip(*toy_obstacle), c='green')
ax.scatter(*zip(*inter), c='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('T')
plt.show()