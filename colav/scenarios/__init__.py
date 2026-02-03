"""
Scenario simulation and testing framework for maritime collision avoidance.

Provides complete simulation environment and visualization tools for testing
and demonstrating the timespace collision avoidance framework with realistic
maritime scenarios including multiple vessels, static obstacles, and COLREGS.

Key Components
--------------
COLAVEnv : Simulation environment with vessels and obstacles
ScenarioRunner : Scenario execution and visualization engine

Examples
--------
Basic scenario setup:

>>> from colav.scenarios import COLAVEnv, ScenarioRunner
>>> from colav.obstacles import MovingShip
>>> from colav.path.pwl import PWLPath
>>> 
>>> # Create environment with own ship and obstacles
>>> env = COLAVEnv(
...     own_ship=MovingShip.from_body((0, 0), 45, 3, 0, 10, 3),
...     path=PWLPath([(0, 0), (100, 100)]),
...     desired_speed=5.0,
...     obstacles=[ship1, ship2],
...     colregs=True
... )
>>> 
>>> # Run and visualize scenario
>>> runner = ScenarioRunner(env, tf=300, dt=5)
>>> runner.run(xlim=(-50, 150), ylim=(-50, 150))
"""

__version__ = "0.1.0"

# Import main modules
from .env import *
from .runner import *