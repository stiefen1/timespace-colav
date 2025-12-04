from colav.scenarios.env import COLAVEnv
from colav.path.planning import PWLPath
from colav.obstacles.moving import MovingShip, MovingObstacle
import tqdm, numpy as np, matplotlib.pyplot as plt, logging, imageio
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
logger = logging.getLogger(__name__)


class ScenarioRunner:
    def __init__(
        self,
        env: COLAVEnv,
        tf: float,
        dt: float = 1, # seconds
    ):
        self.env = env
        self.tf = tf
        self.dt = dt

    def run(self, xlim: Tuple[float, float], ylim: Tuple[float, float], output_file: str = 'simulation.gif', track_own_ship: bool = False) -> None:
        plt.ioff()
        fig, ax = plt.subplots()
        frames = []
        times = []
        distances = []
        own_traj = []
        obs_trajs = []

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

            if not own_traj:
                own_traj = [own_ship.position]
                obs_trajs = [[obs.position] for obs in obstacles]

            # Plot trajectories
            if own_traj:
                ax.plot(*zip(*own_traj), c='blue', linestyle='-')
            for traj in obs_trajs:
                if traj:
                    ax.plot(*zip(*traj), c='red', linestyle='-')

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

            
        logger.info(f"Scenario done. Saving {output_file} ..")
        imageio.mimsave(output_file, frames, fps=10)
        plt.close(fig)

        # Create distance plot
        if distances:
            fig2, ax2 = plt.subplots()
            for i in range(len(distances[0])):
                obs_dists = [d[i] for d in distances]
                ax2.plot(times, obs_dists, label=f"Obstacle {i+1}")
            ax2.axhline(y=self.env.buffer, color='red', linestyle='--', label='Min Acceptable Distance')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.set_title('Distance to Obstacles Over Time')
            fig2.savefig('distance_plot.png')
            plt.close(fig2)
            logger.info("Saved distance_plot.png")

            

if __name__ == "__main__":
    import colav
    colav.configure_logging(logging.WARNING)

    env = COLAVEnv(
            MovingShip.from_body(
                (0, 0),
                45,
                3,
                0,
                10,
                3,
                degrees=True
            ),
            PWLPath(
                [
                    (0, 0),
                    (1000, 1000)
                ]
            ),
            3,
            [
                MovingShip.from_body(
                    (400, 0),
                    -20,
                    2.5,
                    0,
                    10,
                    3,
                    degrees=True
                ),
                MovingShip.from_body(
                    (600, 1000),
                    -180,
                    2,
                    0,
                    10,
                    3,
                    degrees=True
                )
            ],
            max_speed=3.5,
            max_yaw_rate=1,
            colregs=False,
            degrees=True,
            buffer=100,
            simplify=5,
            max_iter=20,
            distance_threshold=1000,
            lookahead_distance=500
        )
    
    runner = ScenarioRunner(
        env, 
        tf=1000,
        dt=2
    )

    runner.run(xlim=(-100, 1100), ylim=(-100, 1100), track_own_ship=False)
    # runner.run(xlim=(-100, 100), ylim=(-100, 100), track_own_ship=True)