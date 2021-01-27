'''
3D quadcopter rendering using matplotlib

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import time
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image


class _Vehicle:

    VEHICLE_SIZE = 0.5
    PROPELLER_RADIUS = 0.2
    PROPELLER_OFFSET = 0.01

    def __init__(self, ax, showtraj, color='b'):

        self.ax_traj = _Vehicle.create_axis(ax, color)

        self.ax_arms = [_Vehicle.create_axis(ax, color) for j in range(4)]
        self.ax_props = [_Vehicle.create_axis(ax, color) for j in range(4)]

        # Support plotting trajectories
        self.showtraj = showtraj

        # Initialize arrays that we will accumulate to plot trajectory
        self.xs = []
        self.ys = []
        self.zs = []

        # For render() support
        self.fig = None

    def update(self, pose):

        x, y, z, phi, theta, psi = pose

        # Adjust for X axis orientation
        theta = -theta

        # Adjust coordinate frame NED => NEU
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
            dy = 2 * (j % 2) - 1

            self._set_axis(x, y, z,
                           phi, theta, psi,
                           self.ax_arms[j],
                           dx*rs, dy*rs, 0)

            self._set_axis(x, y, z,
                           phi, theta, psi,
                           self.ax_props[j],
                           dx*v2+px, dy*v2+py, self.PROPELLER_OFFSET)

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
        a11 = cth*cps
        a12 = sph*sth*cps + cph*sps
        a21 = -cth*sps
        a22 = -sph*sth*sps + cph*cps
        a31 = sth
        a32 = -sph*cth

        # Rotate coordinates
        xx = a11 * xs + a12 * ys
        yy = a21 * xs + a22 * ys
        zz = a31 * xs + a32 * ys

        # Set axis points
        axis.set_data(x+xx, y+yy)
        axis.set_3d_properties(z+zz+dz)

    @staticmethod
    def create_axis(ax, color):
        obj = ax.plot([], [], [], '-', c=color)[0]
        obj.set_data([], [])
        return obj


class ThreeDRenderer:
    '''
    Base class for 3D rendering
    '''

    def __init__(self,
                 env,
                 view_width=1,
                 lim=50,
                 fps=50,
                 label=None,
                 showtraj=False,
                 viewangles=(30, 120),
                 outfile=None):

        # Environment will share position with renderer
        self.env = env

        # We also support different frame rates
        self.fps = fps

        self.radius = env.TARGET_RADIUS

        # Helps us handle window close
        self.is_open = True

        # Set up figure & 3D axis for animation
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, view_width, 1],
                                    projection='3d')

        # Set up axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Set view angles if indicated
        if viewangles is not None:
            self.ax.view_init(*viewangles)

        # Set up formatting for the movie files
        self.writer = None
        self.outfile = outfile
        if self.outfile is not None:
            Writer = animation.writers['ffmpeg']
            self.writer = Writer(fps=15,  # works better than self.fps
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)

        # Set title to name of environment
        self.ax.set_title(label)

        # Set axis limits
        self.ax.set_xlim((-lim, lim))
        self.ax.set_ylim((-lim, lim))
        self.ax.set_zlim((0, lim))

        # Create a representation of the copter
        self.copter = _Vehicle(self.ax, showtraj)

    def start(self):

        # Instantiate the animator
        interval = int(1000/self.fps)
        anim = animation.FuncAnimation(self.fig,
                                       self._animate,
                                       interval=interval,
                                       blit=False)

        # Support window close
        self.fig.canvas.mpl_connect('close_event', self._handle_close)

        # Set up to save a movie if indicated
        if self.outfile is not None:
            anim.save(self.outfile, writer=self.writer)

        # Otherwise, show the display window
        else:
            try:
                plt.show()
            except Exception:
                pass

    def close(self):

        time.sleep(1)
        plt.close(self.fig)
        exit(0)

    def render(self):

        if self.env.done():
            self.close()

        self.copter.update(self.env.pose)

    def _complete(self):

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        buf = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        buf.shape = w, h, 3

        return np.array(Image.frombytes("RGB", (w, h), buf.tostring()))

    def _handle_close(self, event):

        self.is_open = False

    def _animate(self, _):

        # Update the copter animation with vehicle pose
        self.render()

        try:

            # Draw everything
            self.fig.canvas.draw()

        except Exception:
            pass


class ThreeDLanderRenderer(ThreeDRenderer):
    '''
    Extends 3D rendering base class by displaying a landing target
    '''

    def __init__(self, env, viewangles=None, outfile=None, view_width=1):

        ThreeDRenderer.__init__(self,
                                env,
                                lim=10,
                                label='Lander',
                                viewangles=viewangles,
                                outfile=outfile,
                                view_width=view_width)

        self.target_axis = _Vehicle.create_axis(self.ax, 'r')

        self.target_x = env.target[0, :]
        self.target_y = env.target[1, :]
        self.target_z = np.zeros(self.target_x.shape)

    def render(self):

        ThreeDRenderer.render(self)

        # Draw target on ground
        self.target_axis.set_data(self.target_x, self.target_y)
        self.target_axis.set_3d_properties(self.target_z)

        return ThreeDRenderer._complete(self)


class ThreeDVisualLanderRenderer(ThreeDLanderRenderer):
    '''
    Extends 3D landing-target rendering class with a visual display of the
    target
    '''

    MARGIN = 20

    def __init__(self, env, viewangles=None, outfile=None, view_width=1):

        ThreeDLanderRenderer.__init__(self, env, viewangles, outfile,
                                      view_width=0.5)

        self.vision_axis = self.fig.add_axes([0.5, 0, 0.5, 1],
                                             frame_on=False,
                                             aspect='equal',
                                             xticks=[],
                                             xticklabels=[],
                                             yticks=[],
                                             yticklabels=[])

        # Widen the figure
        figsize = self.fig.get_size_inches()
        self.fig.set_size_inches(1.5*figsize[0], figsize[1])

        # Make a red-on-white colormap
        self.cmap = ListedColormap([[1, 1, 1, 1],  [1, 0, 0, 1]])

        # Store image sizes
        self.res = self.env.RESOLUTION
        self.shape = (self.res + 2*self.MARGIN,)*2
        self.lo = self.MARGIN
        self.hi = self.MARGIN + self.res

    def render(self):

        ThreeDLanderRenderer.render(self)

        target = self.env.get_target_image_points()

        self.vision_axis.scatter(target[0, :], target[1, :], c='r', s=2.0)


# End of ThreeDRenderer classes -----------------------------------------------


def make_parser():
    '''
    Exported function to support command-line parsing in scripts.
    You can add your own arguments, then call parse() to get args.
    '''
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--view', required=False, default='30,120',
                        help='View elevation, azimuth')
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--visual', action='store_true',
                        help='Run visual environment')
    return parser


def parse(parser):
    args = parser.parse_args()
    viewangles = tuple((int(s) for s in args.view.split(',')))
    return args, viewangles
