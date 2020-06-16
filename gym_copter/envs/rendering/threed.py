'''
3D quadcopter rendering using matplotlib

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

        self.ax_traj  = create_axis(ax, color)

        self.ax_arms   = [create_axis(ax, color) for j in range(4)]
        self.ax_props  = [create_axis(ax, color) for j in range(4)]

        # Support plotting trajectories
        self.showtraj = showtraj

        # Initialize arrays that we will accumulate to plot trajectory
        self.xs = []
        self.ys = []
        self.zs = []

    def update(self, x, y, z, phi, theta, psi):

        # Adjust coordinate frame
        x = -x
        z = -z

        # Append position to arrays for plotting trajectory
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)

        # Plot trajectory if indicated
        if self.showtraj:
            self.ax_traj.set_data(self.xs, self.ys)
            self.ax_traj.set_3d_properties(self.zs)

        # Create points for arms
        v2 = self.VEHICLE_SIZE / 2
        rs = np.linspace(0, v2)

        # Create points for propellers
        px = self.PROPELLER_RADIUS * np.sin(np.linspace(-np.pi, +np.pi))
        py = self.PROPELLER_RADIUS * np.cos(np.linspace(-np.pi, +np.pi))

        # Loop over arms and propellers
        for j in range(4):

            dx = 2 * (j // 2) - 1
            dy = 2 * (j %  2) - 1

            self._set_axis(x, y, z, phi, theta, psi, self.ax_arms[j], dx*rs, dy*rs, 0)

            self._set_axis(x, y, z, phi, theta, psi, self.ax_props[j], dx*v2+px, dy*v2+py, self.PROPELLER_OFFSET)

        plt.gca().set_aspect('equal')

    def _set_axis(self, x, y, z, phi, theta, psi, axis, xs, ys, dz):

        # Make convenient abbreviations for functions of Euler angles
        cph = np.cos(phi)
        sph = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cps = np.cos(psi)
        sps = np.sin(psi)

        # Build rotation matrix: 
        # see http://www.kwon3d.com/theory/euler/euler_angles.html, Eqn. 2
        a11 =  cth*cps
        a12 =  sph*sth*cps + cph*sps
        a21 = -cth*sps 
        a22 = -sph*sth*sps + cph*cps
        a31 =  sth
        a32 = -sph*cth

        # Rotate coordinates
        xx = a11 * xs + a12 * ys 
        yy = a21 * xs + a22 * ys
        zz = a31 * xs + a32 * ys

        # Set axis points
        axis.set_data(x+xx, y+yy)
        axis.set_3d_properties(z+zz+dz)

class ThreeDRenderer:

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
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.set_zlim((0, lim))

        # Create a representation of the copter
        self.copter = _Vehicle(self.ax, showtraj)

    def start(self):

        # Instantiate the animator
        anim = animation.FuncAnimation(self.fig, self._animate, interval=int(1000/self.env.FRAMES_PER_SECOND), blit=False)
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

        # Update the copter animation with vehicle pose
        self.copter.update(*self.env.pose)

        # Draw everything
        self.fig.canvas.draw()
