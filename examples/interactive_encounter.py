import matplotlib.pyplot as plt, numpy as np, os
from matplotlib.widgets import Slider
from colav.obstacles.moving import MovingShip
from colav.colregs.encounters import get_recommendation_for_os, get_encounter, Encounter


# def clear_terminal():
#     os.system('cls' if os.name == 'nt' else 'clear')

class InteractiveEncounter:
    def __init__(self):
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.3, right=0.7)
        
        # Initial parameters
        self.os_x = 10
        self.os_y = 10
        self.os_heading = 45
        self.ts_x = 30
        self.ts_y = 20
        self.ts_heading = 225
        
        # Create ships
        self.os = MovingShip((self.os_x, self.os_y), self.os_heading, (0, 0), loa=15, beam=4, degrees=True)
        self.ts = MovingShip((self.ts_x, self.ts_y), self.ts_heading, (0, 0), loa=10, beam=3, degrees=True)
        
        # Create sliders
        self.create_sliders()
        
        # Initial plot
        self.update_plot()
        
    def create_sliders(self):
        """Create all the slider widgets."""
        # Slider positions
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.15
        slider_width = 0.5
        
        # OS sliders
        ax_os_x = plt.axes((slider_left, 0.25, slider_width, slider_height))
        ax_os_y = plt.axes((slider_left, 0.25 - slider_spacing, slider_width, slider_height))
        ax_os_heading = plt.axes((slider_left, 0.25 - 2*slider_spacing, slider_width, slider_height))
        
        # TS sliders
        ax_ts_x = plt.axes((slider_left, 0.25 - 4*slider_spacing, slider_width, slider_height))
        ax_ts_y = plt.axes((slider_left, 0.25 - 5*slider_spacing, slider_width, slider_height))
        ax_ts_heading = plt.axes((slider_left, 0.25 - 6*slider_spacing, slider_width, slider_height))
        
        # Create sliders
        self.slider_os_x = Slider(ax_os_x, 'OS X', 0, 50, valinit=self.os_x)
        self.slider_os_y = Slider(ax_os_y, 'OS Y', 0, 40, valinit=self.os_y)
        self.slider_os_heading = Slider(ax_os_heading, 'OS Heading', 0, 360, valinit=self.os_heading)
        
        self.slider_ts_x = Slider(ax_ts_x, 'TS X', 0, 50, valinit=self.ts_x)
        self.slider_ts_y = Slider(ax_ts_y, 'TS Y', 0, 40, valinit=self.ts_y)
        self.slider_ts_heading = Slider(ax_ts_heading, 'TS Heading', 0, 360, valinit=self.ts_heading)
        
        # Connect sliders to update function
        self.slider_os_x.on_changed(self.update_parameters)
        self.slider_os_y.on_changed(self.update_parameters)
        self.slider_os_heading.on_changed(self.update_parameters)
        self.slider_ts_x.on_changed(self.update_parameters)
        self.slider_ts_y.on_changed(self.update_parameters)
        self.slider_ts_heading.on_changed(self.update_parameters)
    
    def update_parameters(self, val):
        """Update parameters from sliders and refresh plot."""
        self.os_x = self.slider_os_x.val
        self.os_y = self.slider_os_y.val
        self.os_heading = self.slider_os_heading.val
        self.ts_x = self.slider_ts_x.val
        self.ts_y = self.slider_ts_y.val
        self.ts_heading = self.slider_ts_heading.val
        
        # Update ship states
        self.os = MovingShip((self.os_x, self.os_y), self.os_heading, (0, 0), loa=15, beam=4, degrees=True)
        self.ts = MovingShip((self.ts_x, self.ts_y), self.ts_heading, (0, 0), loa=10, beam=3, degrees=True)
        
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current ship positions and encounter analysis."""
        self.ax.clear()
        
        # Plot ships
        self.os.plot(ax=self.ax, c='blue', alpha=0.8)
        self.ts.plot(ax=self.ax, c='red', alpha=0.8)
        
        # Add ship labels
        self.ax.text(self.os_x + 2, self.os_y + 2, 'OS (Own Ship)', fontsize=12, c='blue', fontweight='bold')
        self.ax.text(self.ts_x + 2, self.ts_y + 2, 'TS (Target Ship)', fontsize=12, c='red', fontweight='bold')
        
        # Draw line between ships
        self.ax.plot([self.os_x, self.ts_x], [self.os_y, self.ts_y], 'k--', alpha=0.5, linewidth=1)

        # # Recommendation system
        reco, info = get_recommendation_for_os(self.os, self.ts)
        encounter_os_vs_ts = info['encounter_ts_wrt_os']
        encounter_ts_vs_os = info['encounter_os_wrt_ts']

        # Calculate distance and bearing
        dx = self.ts_x - self.os_x
        dy = self.ts_y - self.os_y
        distance = np.sqrt(dx**2 + dy**2)
        bearing = np.degrees(np.arctan2(dx, dy))
        
        # Add encounter information as text
        info_text = f"Distance: {distance:.1f} units\n"
        info_text += f"Bearing (OS→TS): {bearing:.1f}°\n\n"
        info_text += f"OS vs TS: {encounter_os_vs_ts.name}\n"
        info_text += f"TS vs OS: {encounter_ts_vs_os.name}\n\n"
        info_text += f"OS Heading: {self.os_heading:.1f}°\n"
        info_text += f"TS Heading: {self.ts_heading:.1f}°"
        
        # Color code the encounter types
        encounter_colors = {
            Encounter.HEAD_ON: 'orange',
            Encounter.STARBOARD: 'green',
            Encounter.PORT: 'purple',
            Encounter.OVERTAKING: 'brown',
            Encounter.INVALID: 'gray'
        }
        
        # Add text box with encounter information
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add colored indicators for encounter types
        os_color = encounter_colors.get(encounter_os_vs_ts, 'black')
        ts_color = encounter_colors.get(encounter_ts_vs_os, 'black')
        
        self.ax.plot(self.os_x, self.os_y, 'o', c=os_color, markersize=15, alpha=0.7)
        self.ax.plot(self.ts_x, self.ts_y, 'o', c=ts_color, markersize=15, alpha=0.7)
        
        # Set axis properties
        self.ax.set_xlim(-5, 55)
        self.ax.set_ylim(-5, 45)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_title('Interactive COLREGS Encounter Analysis')
        
        # Add legend for encounter types
        legend_elements = []
        for encounter, color in encounter_colors.items():
            if encounter != Encounter.INVALID:
                legend_elements.append(plt.Line2D([0], [0], marker='o', c='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=encounter.name))
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Refresh the plot
        self.fig.canvas.draw()
    
    def show(self):
        """Display the interactive figure."""
        plt.show()


if __name__ == "__main__":
    import colav, logging
    colav.configure_logging(logging.INFO)
    InteractiveEncounter().show()
