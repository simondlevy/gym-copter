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

from threading import Thread

def lorentz_deriv(xyz, t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    x,y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


class TPV:

    def __init__(self, env):

        self.env = env

        self.is_open = True

        # Set up figure & 3D axis for animation
        self.fig = plt.figure()
        ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(env.unwrapped.spec.id)

        # Choose random starting points, uniformly distributed from -15 to 15
        np.random.seed(1)
        x0 = -15 + 30 * np.random.random((1, 3))

        # Solve for the trajectories
        t = np.linspace(0, 4, 1000)
        self.x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t) for x0i in x0])

        # prepare the axes limits
        ax.set_xlim((-100, 100))
        ax.set_ylim((-100, 100))
        ax.set_zlim((-100, 100))

        # set point-of-view: specified by (altitude degrees, azimuth degrees)
        ax.view_init(30, 0)

        # set up line and point
        self.line = ax.plot([], [], [], '-', c='b')[0]
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        self.pt   = ax.plot([], [], [], 'o', c='b')[0]
        self.pt.set_data([], [])
        self.pt.set_3d_properties([])

    def start(self):

        # instantiate the animator.
        anim = animation.FuncAnimation(self.fig, self._animate, frames=500, interval=30, blit=False)
        self.fig.canvas.mpl_connect('close_event', self._handle_close)

        plt.show()

    def _handle_close(self, event):

        self.is_open = False
        
    def _animate(self, i):

        x,y,z = self.env.state[0:6:2]

        print('%+3.3f %+3.3f %+3.3f' % (x,y,z))

        self.pt.set_data(x, y)
        self.pt.set_3d_properties(z)

        '''
        i = (2 * i) % self.x_t.shape[1]

        for xi in self.x_t:
            x, y, z = xi[:i].T

            print(x,y,z,'\n')

            self.line.set_data(x, y)
            self.line.set_3d_properties(z)

            self.pt.set_data(x[-1:], y[-1:])
            self.pt.set_3d_properties(z[-1:])
        '''

        self.fig.canvas.draw()
