'''
3D quadcopter rendering using matplotlib

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import mpl_toolkits.mplot3d.art3d as art3d
from PIL import Image


def _create_line3d(axes, color):
    '''
    Helper to create object for erasable plotting
    '''
    line3d = axes.plot([], [], [], '-', c=color)[0]
    line3d.set_data([], [])
    return line3d


class _Vehicle:

    VEHICLE_SIZE = 0.5
    PROPELLER_RADIUS = 0.2
    PROPELLER_OFFSET = 0.01

    def __init__(self, ax, showtraj, color='b'):

        self.traj_line = _create_line3d(ax, color)

        self.arms_lines = [_create_line3d(ax, color) for j in range(4)]
        self.props_lines = [_create_line3d(ax, color) for j in range(4)]

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
            self.traj_line.set_data(self.xs, self.ys)
            self.traj_line.set_3d_properties(self.zs)

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

            self._set_axes(x, y, z,
                           phi, theta, psi,
                           self.arms_lines[j],
                           dx*rs, dy*rs, 0)

            self._set_axes(x, y, z,
                           phi, theta, psi,
                           self.props_lines[j],
                           dx*v2+px, dy*v2+py, self.PROPELLER_OFFSET)

    def _set_axes(self, x, y, z, phi, theta, psi, axis, xs, ys, dz):

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
        self.open = True

        # Set up figure & 3D axis for animation
        self.fig = plt.figure()
        self.axes = self.fig.add_axes([0, 0, view_width, 1], projection='3d')

        # Set up axis labels
        self.axes.set_xlabel('X (m)')
        self.axes.set_ylabel('Y (m)')
        self.axes.set_zlabel('Z (m)')

        # Set view angles if indicated
        if viewangles is not None:
            self.axes.view_init(*viewangles)

        # Set up formatting for the movie files
        self.writer = None
        self.outfile = outfile
        if self.outfile is not None:
            Writer = animation.writers['ffmpeg']
            self.writer = Writer(fps=15,  # works better than self.fps
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)

        # Set title to name of environment
        self.axes.set_title(label)

        # Set axis limits
        self.axes.set_xlim((-lim, lim))
        self.axes.set_ylim((-lim, lim))
        self.axes.set_zlim((0, lim))

        # Create a representation of the copter
        self.copter = _Vehicle(self.axes, showtraj)

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

    def is_open(self):

        return self.open

    def _complete(self):

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        buf = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        buf.shape = w, h, 3

        return np.array(Image.frombytes("RGB", (w, h), buf.tostring()))

    def _handle_close(self, event):

        self.open = False

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
    # Number of target points (arbitrary)
    TARGET_POINTS = 250

    def __init__(self, env, viewangles=None, outfile=None, view_width=1):

        ThreeDRenderer.__init__(self,
                                env,
                                lim=10,
                                label='Lander',
                                viewangles=viewangles,
                                outfile=outfile,
                                view_width=view_width)

        p = Circle((0, 0), env.TARGET_RADIUS, color='gray', alpha=0.5)
        self.axes.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, zdir='z')

        # Create points for landing zone
        pts = np.linspace(-np.pi, +np.pi, self.TARGET_POINTS)
        target = np.array([np.sin(pts), np.cos(pts)]).transpose()

        self.target_x = target[:, 0]
        self.target_y = target[:, 1]
        self.target_z = np.zeros(len(self.target_x))

    def render(self):

        ThreeDRenderer.render(self)

        return ThreeDRenderer._complete(self)
