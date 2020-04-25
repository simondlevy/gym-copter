#!/usr/bin/env python3
'''
Adapted from

https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/
'''

import time
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

N_trajectories = 1

def lorentz_deriv(xyz, t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    x,y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


class TPV:

    def __init__(self):

        self.state = None
        self.is_open = True

        # Set up figure & 3D axis for animation
        self.fig = plt.figure()
        ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')

        # Setting the axes properties
        ax.set_xlim3d([0.0, 1.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, 1.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 1.0])
        ax.set_zlabel('Z')

        ax.set_title('3D Test')

        # Choose random starting points, uniformly distributed from -15 to 15
        np.random.seed(1)
        x0 = -15 + 30 * np.random.random((N_trajectories, 3))

        # Solve for the trajectories
        t = np.linspace(0, 4, 1000)
        self.x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t) for x0i in x0])

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

        # set up lines and points
        self.lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
        self.pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

        # prepare the axes limits
        ax.set_xlim((-25, 25))
        ax.set_ylim((-35, 35))
        ax.set_zlim((5, 55))

        # set point-of-view: specified by (altitude degrees, azimuth degrees)
        ax.view_init(30, 0)

        # instantiate the animator.
        anim = animation.FuncAnimation(self.fig, self._animate, init_func=self._init, frames=500, interval=30, blit=False)

        self.fig.canvas.mpl_connect('close_event', self._handle_close)

        plt.show()

    def display(self, mode, state):

        self.state = state

    def isOpen(self):

        return self.is_open

    def _handle_close(self, event):

        self.is_open = False
        
    def _animate(self, i):

        # we'll step two time-steps per frame.  This leads to nice results.
        i = (2 * i) % self.x_t.shape[1]

        for line, pt, xi in zip(self.lines, self.pts, self.x_t):
            x, y, z = xi[:i].T
            line.set_data(x, y)
            line.set_3d_properties(z)

            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        #ax.view_init(30, 0.3 * i)
        self.fig.canvas.draw()
        return self.lines + self.pts

    # initialization function: plot the background of each frame
    def _init(self):
        for line, pt in zip(self.lines, self.pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return self.lines + self.pts


