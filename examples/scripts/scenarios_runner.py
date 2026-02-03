#!/usr/bin/env python3
"""
Simple Maritime Collision Avoidance Scenario Demo

Demonstrates the key features of the scenarios module for testing
and visualizing maritime collision avoidance algorithms.

This example creates a realistic encounter scenario with:
- Own ship following a planned path
- Multiple target ships on collision courses  
- COLREGS compliance -> breach to avoid collision
- Safety buffers and performance constraints
- Animated visualization and analysis plots

Run this script to see the collision avoidance framework in action!
"""

import logging

import colav
from colav.scenarios import COLAVEnv, ScenarioRunner
from colav.obstacles import MovingShip
from colav.path.pwl import PWLPath
from shapely import Polygon, Point

def create_crossing_scenario():
    """Create a crossing encounter scenario with COLREGS compliance."""
    
    # Configure logging to see what's happening
    colav.configure_logging(level=logging.INFO)
    
    # Define own ship starting from southwest, heading northeast
    own_ship = MovingShip.from_body(
        position=(-500, -500),    # Start position (m)
        psi=45,                   # Initial heading (45° NE)
        u=8,                      # Forward speed (m/s)  
        v=0,                      # No sideways motion
        loa=50,                   # Length (m)
        beam=8,                   # Width (m)
        degrees=True,
        mmsi=123456789
    )
    
    # Define the planned path - simple straight line to destination
    planned_path = PWLPath([
        (-500, -500),  # Start
        (0, 0),        # Waypoint 
        (500, 500)     # Destination
    ])
    
    # Create target ships on potential collision courses
    target_ships = [
        # Ship 1: Crossing from east to west (starboard crossing)
        MovingShip.from_body(
            position=(200, 0),
            psi=245,              # Heading west
            u=6,                  # 6 m/s
            v=0,
            loa=40, beam=7,
            degrees=True,
            mmsi=987654321,
            du=0.2,               # Small speed uncertainty
            dchi=3                # Small heading uncertainty (deg)
        ),
        
        # Ship 2: Head-on encounter from northeast
        MovingShip.from_body(
            position=(300, 300),
            psi=225,              # Heading southwest (opposite direction)
            u=7,                  # 7 m/s
            v=0,
            loa=35, beam=6,
            degrees=True,
            mmsi=456789123,
            du=0.3,
            dchi=5
        )
    ]
    
    # Add some static obstacles (islands or restricted areas)
    shore_obstacles = [
        # Small island
        Point(100, 200).buffer(30),
        
        # Rectangular restricted area
        Polygon([(100, -200), (250, -200), (250, -100), (100, -100)])
    ]
    
    # Create the simulation environment
    env = COLAVEnv(
        own_ship=own_ship,
        path=planned_path,
        desired_speed=8.0,              # Target speed for planning
        obstacles=target_ships,
        shore=shore_obstacles,
        
        # Collision avoidance parameters
        max_speed=8.0,                  # Maximum allowed speed
        max_course_rate=1,              # Max turn rate (deg/s)
        colregs=True,                   # Enable COLREGS compliance
        
        # Safety and performance settings
        distance_threshold=1500,        # Activate COLAV within 1.5km
        lookahead_distance=500,         # Plan 400m ahead on path
        buffer_moving=30,               # 25m safety buffer around ships
        buffer_static=30,               # 10m buffer around static obstacles
        simplify_static=20,
        
        # Optimization parameters
        max_iter=20,                    # Max planning iterations
        speed_factor=0.9,               # Speed reduction factor
        abort_colregs_after_iter=5     # Relax COLREGS if needed
    )
    
    return env

def run_scenario_demo():
    """Run the complete scenario demonstration."""
    
    print("Maritime Collision Avoidance Demo")
    print("=" * 40)
    
    # Create the scenario
    print("Setting up crossing encounter scenario...")
    env = create_crossing_scenario()
    
    # Create the scenario runner
    runner = ScenarioRunner(
        env=env,
        tf=400,                        # Run for 400 seconds
        dt=2                           # 2-second time steps
    )
    
    print("Running simulation with visualization...")
    print("This will generate:")
    print("- simulation.gif: Animated scenario")
    print("- distance_plot.png: Safety distance analysis")  
    print("- speed_course_plot.png: Performance metrics")
    
    # Run with fixed view to see the whole scenario
    runner.run(
        xlim=(-600, 600),              # X-axis range (m)
        ylim=(-600, 600),              # Y-axis range (m)
        output_file='collision_avoidance_demo.gif',
        track_own_ship=False           # Fixed view
    )
    
    print("\nSimulation complete!")
    print("Check the generated files to analyze the collision avoidance behavior.")
    
def run_ship_tracking_demo():
    """Run a second demo with ship-centered view."""
    
    print("\nRunning ship-tracking view demo...")
    
    env = create_crossing_scenario()
    runner = ScenarioRunner(env, tf=200, dt=2)
    
    # Run with ship-tracking view for closer analysis
    runner.run(
        xlim=(-200, 200),              # ±200m from own ship
        ylim=(-200, 200),
        output_file='ship_tracking_view.gif',
        track_own_ship=True            # Camera follows own ship
    )
    
    print("Ship-tracking demo complete!")

if __name__ == "__main__":
    try:
        # Run the main demonstration
        run_scenario_demo()
        
        # Ask user if they want the ship-tracking demo too
        response = input("\nRun ship-tracking view demo? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            run_ship_tracking_demo()
            
        print("\nAll demos complete! Check the generated files:")
        print("- collision_avoidance_demo.gif")
        print("- distance_plot.png") 
        print("- speed_course_plot.png")
        if response in ['y', 'yes']:
            print("- ship_tracking_view.gif")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError running demo: {e}")
        logging.error(f"Demo failed: {e}", exc_info=True)
