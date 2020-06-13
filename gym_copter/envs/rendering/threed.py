'''
Third-Person (3D) view using matplotlib

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def create_axis(ax, color):
    obj = ax.plot([], [], [], '-', c=color)[0]
    obj.set_data([], [])
    obj.set_3d_properties([])
    return obj

class _Vehicle:

    VEHICLE_SIZE      = 0.5
    PROPELLER_RADIUS  = 0.2
    PROPELLER_OFFSET  = 0.01

    def __init__(self, ax, showtraj, color='b'):

        # Set up line and point
        self.ax_traj  = create_axis(ax, color)
        self.ax_arm1  = create_axis(ax, color)
        self.ax_arm2  = create_axis(ax, color)
        self.ax_prop1 = create_axis(ax, color)
        self.ax_prop2 = create_axis(ax, color)
        self.ax_prop3 = create_axis(ax, color)
        self.ax_prop4 = create_axis(ax, color)

        # Support plotting trajectories
        self.showtraj = showtraj

        # Initialize arrays that we will accumulate to plot trajectory
        self.xs = []
        self.ys = []
        self.zs = []

    def update(self, x, y, z):

        # Append position to arrays for plotting trajectory
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)

        # Plot trajectory if indicated
        if self.showtraj:
            self.ax_traj.set_data(self.xs, self.ys)
            self.ax_traj.set_3d_properties(self.zs)

        # Show vehicle
        d = self.VEHICLE_SIZE / 2
        xs = x + np.linspace(-d, +d)
        ys = y + np.linspace(-d, +d)
        zs = z * np.ones(xs.shape)
        self.ax_arm1.set_data(xs, ys)
        self.ax_arm1.set_3d_properties(zs)
        self.ax_arm2.set_data(xs, -ys)
        self.ax_arm2.set_3d_properties(zs)
        xs = x + d + self.PROPELLER_RADIUS * np.sin(np.linspace(-np.pi, +np.pi))
        ys = y + d + self.PROPELLER_RADIUS * np.cos(np.linspace(-np.pi, +np.pi))
        self.ax_prop1.set_data(xs, ys)
        self.ax_prop1.set_3d_properties(zs+self.PROPELLER_OFFSET)

class ThreeD:

    def __init__(self, env, lim=50, label=None, showtraj=False, viewangles=None):

        # Environment will be used to get position
        self.env = env

        # Helps us handle window close
        self.is_open = True

        # Set up figure & 3D axis for animation
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')

        # Set up axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        if viewangles is not None:
            self.ax.view_init(*viewangles)

        # Set title to name of environment
        self.ax.set_title(env.unwrapped.spec.id if label is None else label)

        # Set axis limits
        lim = 5
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.set_zlim((0, lim))

        # Create a representation of the copter
        self.copter = _Vehicle(self.ax, showtraj)

    def start(self):

        # Instantiate the animator
        anim = animation.FuncAnimation(self.fig, self._animate, interval=int(1000/self.env.FPS), blit=False)
        self.fig.canvas.mpl_connect('close_event', self._handle_close)

        # Show the display window
        try:
            plt.show()
        except:
            pass

    def close(self):

        time.sleep(1)
        plt.close(self.fig)

    def _handle_close(self, event):

        self.is_open = False
        
    def _animate(self, _):

        # Get vehicle position
        x,y,z,_,_ = self.env.pose
        
        # Update the copter animation
        self.copter.update(x, y, z)

        # Draw everything
        self.fig.canvas.draw()
