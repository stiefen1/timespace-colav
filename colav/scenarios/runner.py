"""
Scenario execution and visualization framework.

Provides comprehensive scenario runner for executing collision avoidance
simulations over time with real-time visualization, data collection,
and analysis output for maritime navigation scenarios.

Key Components
--------------
ScenarioRunner : Complete scenario execution and visualization engine

Notes
-----
Generates animated visualizations, distance plots, and performance metrics
for comprehensive analysis of collision avoidance behavior and compliance.
"""

from colav.scenarios.env import COLAVEnv
from colav.path.pwl import PWLPath, PWLTrajectory
from colav.obstacles.moving import MovingShip, MovingObstacle
import tqdm, numpy as np, matplotlib.pyplot as plt, logging, imageio
from typing import List, Tuple
from shapely import Polygon, Point
import matplotlib, os
matplotlib.use('Agg')
logger = logging.getLogger(__name__)

class ScenarioRunner:
    """
    Scenario execution and visualization engine.
    
    Runs collision avoidance scenarios over time with comprehensive
    visualization including animated plots, distance tracking, speed
    profiles, and course rate analysis for maritime navigation testing.
    
    Parameters
    ----------
    env : COLAVEnv
        Configured simulation environment with vessels and obstacles.
    tf : float
        Final simulation time in seconds.
    dt : float, default 1
        Time step size in seconds for simulation updates.
        
    Attributes
    ----------
    env : COLAVEnv
        The simulation environment being executed.
    tf : float
        Total simulation duration.
    dt : float
        Simulation time step.
        
    Methods
    -------
    run(xlim, ylim, **kwargs)
        Execute complete scenario with visualization
        
    Examples
    --------
    Basic scenario execution:
    
    >>> runner = ScenarioRunner(env, tf=300, dt=5)
    >>> runner.run(
    ...     xlim=(-100, 200),
    ...     ylim=(-100, 200),
    ...     output_file='scenario.gif'
    ... )
    
    Tracking own ship view:
    
    >>> runner.run(
    ...     xlim=(-50, 50),    # Relative to own ship
    ...     ylim=(-50, 50),
    ...     track_own_ship=True
    ... )
    
    Notes
    -----
    Visualization features:
    - Animated GIF showing vessel movements and trajectories
    - Real-time collision avoidance trajectory display
    - Distance-to-obstacles tracking plots
    - Speed and course rate performance analysis
    - Static obstacle and shore boundary display
    
    Output files generated:
    - simulation.gif: Animated scenario visualization
    - distance_plot.png: Distance vs time for all obstacles
    - speed_course_plot.png: Speed and course rate profiles
    
    Performance metrics tracked:
    - Minimum distances to all obstacles
    - Speed profile vs limits
    - Course rate usage vs constraints
    - Path following accuracy
    - Collision avoidance success
    
    Designed for:
    - Algorithm validation and testing
    - Regulatory compliance verification
    - Performance benchmarking
    - Educational demonstration
    - Research scenario analysis
    
    See Also
    --------
    COLAVEnv : Simulation environment configuration
    TimeSpaceColav : Underlying collision avoidance algorithm
    """
    
    def __init__(
        self,
        env: COLAVEnv,
        tf: float,
        dt: float = 1, # seconds
    ):
        self.env = env
        self.tf = tf
        self.dt = dt

    def run(self, xlim: Tuple[float, float], ylim: Tuple[float, float], output_folder: str = '', output_file: str = 'simulation.gif', track_own_ship: bool = False) -> None:
        """
        Execute complete scenario with visualization and analysis.
        
        Runs the simulation from start to finish, generating animated
        visualization and comprehensive performance analysis including
        distance tracking, speed profiles, and safety metrics.
        
        Parameters
        ----------
        xlim : tuple of float
            X-axis limits for visualization (x_min, x_max) in meters.
            If track_own_ship=True, these are relative to own ship.
        ylim : tuple of float
            Y-axis limits for visualization (y_min, y_max) in meters.
            If track_own_ship=True, these are relative to own ship.
        output_file : str, default 'simulation.gif'
            Filename for animated GIF output.
        track_own_ship : bool, default False
            Whether to center view on own ship (moving camera) or
            use fixed coordinate system.
            
        Notes
        -----
        Execution process:
        1. Initialize visualization and data collection
        2. Run simulation loop with time steps
        3. Collect trajectory, distance, and performance data
        4. Generate animated visualization frames
        5. Create analysis plots and save outputs
        
        Generated outputs:
        - {output_file}: Animated GIF of scenario
        - distance_plot.png: Distance vs time analysis
        - speed_course_plot.png: Performance metrics
        
        Visualization elements:
        - Own ship (blue) with trajectory history
        - Target ships (red) with trajectory histories
        - Planned collision avoidance trajectories
        - Reference path (dashed blue line)
        - Static obstacles and shore (grey)
        - Real-time distance and speed information
        
        Analysis metrics:
        - Minimum separation distances maintained
        - Speed profile vs operational limits
        - Course rate usage vs maneuverability constraints
        - Collision avoidance activation frequency
        - Path following accuracy
        
        Examples
        --------
        Fixed view scenario:
        
        >>> runner.run(
        ...     xlim=(-500, 1500),
        ...     ylim=(-500, 1500),
        ...     output_file='crossing_encounter.gif'
        ... )
        
        Own ship tracking view:
        
        >>> runner.run(
        ...     xlim=(-200, 200),  # ±200m from own ship
        ...     ylim=(-200, 200),
        ...     track_own_ship=True,
        ...     output_file='ship_view.gif'
        ... )
        """
        plt.ioff()
        fig, ax = plt.subplots()
        frames = []
        times = []
        distances = []
        own_traj = []
        obs_trajs = []
        speeds = []
        course_rates = []
        prev_psi = None

        for ti in tqdm.tqdm(np.arange(0, self.tf, self.dt)):
            ax.cla()
            if track_own_ship:
                # Will set limits after getting own_ship position
                pass
            else:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            self.env.path.plot('--', ax=ax, c='blue')
            for static_obs in self.env.shore:
                ax.plot(*static_obs.exterior.xy, c='grey')

            out, done, info = self.env.step(self.dt)
            own_ship: MovingShip = out['own_ship']
            obstacles: List[MovingObstacle] = out['obstacles']
            os_traj: PWLTrajectory = out['trajectory']

            if not own_traj:
                own_traj = [own_ship.position]
                obs_trajs = [[obs.position] for obs in obstacles]

            # Plot trajectories
            if own_traj:
                ax.plot(*zip(*own_traj), c='blue', linestyle='-')
            for traj in obs_trajs:
                if traj:
                    ax.plot(*zip(*traj), c='red', linestyle='-')
            if os_traj is not None:
                os_traj.plot(ax=ax, c='blue', linewidth=0.5)

            if track_own_ship:
                x, y = own_ship.position
                ax.set_xlim(x + xlim[0], x + xlim[1])
                ax.set_ylim(y + ylim[0], y + ylim[1])

            own_ship.plot(ax=ax, c='blue')
            for i, obs in enumerate(obstacles):
                logger.debug(ti)
                obs.plot(ax=ax, c='red', label=f"TS{i+1}")

            # Collect data for distance plot
            times.append(ti)
            dists = [np.linalg.norm(np.array(own_ship.position) - np.array(obs.position)) for obs in obstacles]
            distances.append(dists)

            # Collect speed and course rate
            speed = np.sqrt(own_ship.u**2 + own_ship.v**2)
            speeds.append(speed)
            current_psi = own_ship.psi
            if prev_psi is not None:
                course_rate = (current_psi - prev_psi) / self.dt
                course_rates.append(course_rate)
            else:
                course_rates.append(0)  # For the first point
            prev_psi = current_psi

            # Update trajectories
            own_traj.append(own_ship.position)
            for i, obs in enumerate(obstacles):
                obs_trajs[i].append(obs.position)

            if done:
                logger.info(f"Reached final point of the desired path, ending scenario..")
                break

            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))[:, :, 1:]
            frames.append(frame)

        logger.info(f"Scenario done. Saving {output_file} in {output_folder} ..")
        imageio.mimsave(os.path.join(output_folder, output_file), frames, fps=10)
        plt.close(fig)

        # Create distance plot
        if distances:
            fig2, ax2 = plt.subplots()
            for i in range(len(distances[0])):
                obs_dists = [d[i] for d in distances]
                ax2.plot(times, obs_dists, label=f"Obstacle {i+1}")
            ax2.axhline(y=self.env.buffer_moving, color='red', linestyle='--', label='Min Acceptable Distance')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.set_title('Distance to Obstacles Over Time')
            ax2.set_ylim((0, ax2.get_ylim()[1]))
            fig2.savefig(os.path.join(output_folder, 'distance_plot.png'))
            plt.close(fig2)
            logger.info("Saved distance_plot.png")

        # Create speed and course rate plot
        if speeds and course_rates:
            fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 6))
            ax3.plot(times, speeds)
            ax3.axhline(y=self.env.planner.max_speed, color='green', linestyle='--', label='Max Speed')
            ax3.set_title('Own Ship Speed')
            ax3.set_ylabel('Speed (m/s)')
            ax3.legend()
            ax3.set_ylim((0, self.env.planner.max_speed + 1))
            ax4.plot(times, course_rates)
            ax4.axhline(y=self.env.planner.max_course_rate, color='orange', linestyle='--', label='Max Course Rate')
            ax4.axhline(y=-self.env.planner.max_course_rate, color='orange', linestyle='--', label='Min Course Rate')
            ax4.set_title('Own Ship Course Rate')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Course Rate (rad/s)')
            if self.env.planner.max_course_rate != float('inf'):
                ax4.set_ylim((-self.env.planner.max_course_rate - 1, self.env.planner.max_course_rate + 1))
            ax4.legend()
            fig3.tight_layout()
            fig3.savefig(os.path.join(output_folder, 'speed_course_plot.png'))
            plt.close(fig3)
            logger.info("Saved speed_course_plot.png")

            

if __name__ == "__main__":
    import colav
    colav.configure_logging(logging.ERROR)

    env = COLAVEnv(
            MovingShip.from_body((-200, -200), 45, 3, 0, 10, 3, degrees=True),
            PWLPath([(-200, -200), (200, 400), (600, 500), (1000, 1000)]),
            3,
            [
                MovingShip.from_body((600, 100), -30, 2, 0, 10, 3, degrees=True, du=0.3, dchi=10),
                MovingShip.from_body((1000, 400), -90, 3, 0, 10, 3, degrees=True, du=0.3, dchi=10)
            ],
            shore=[
                Polygon([(800, 600), (800, 700), (700, 700), (700, 600), (800, 600)]),
                Point(200, 0).buffer(100)
            ],
            max_speed=3,
            max_course_rate=0.5, # More restrictive than actual performance of own ship
            colregs=True,
            degrees=True,
            buffer_moving=50,
            simplify_moving=0,
            buffer_static=50,
            simplify_static=5,
            max_iter=50,
            distance_threshold=2000,
            lookahead_distance=1000,
            abort_colregs_after_iter=5,
            speed_factor=0.95
        )
    
    runner = ScenarioRunner(
        env, 
        tf=1000,
        dt=10
    )

    runner.run(xlim=(-300, 1100), ylim=(-300, 1100), track_own_ship=False)
    # runner.run(xlim=(-100, 100), ylim=(-100, 100), track_own_ship=True)