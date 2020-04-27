'''
Third-Person (3D) view using matplotlib

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class _Vehicle:

    def __init__(self, ax, color='b'):

        # Set up line and point
        self.line = ax.plot([], [], [], '-', c=color)[0]
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        self.pt   = ax.plot([], [], [], 'o', c=color)[0]
        self.pt.set_data([], [])
        self.pt.set_3d_properties([])

        # Initialize arrays that we will accumulate to plot trajectory
        self.xs = []
        self.ys = []
        self.zs = []

    def update(self, x, y, z):

        # Append position to arrays for plotting trajectory
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)

        # Plot trajectory
        self.line.set_data(self.xs, self.ys)
        self.line.set_3d_properties(self.zs)

        # Show vehicle as a dot
        self.pt.set_data(x, y)
        self.pt.set_3d_properties(z)

class TPV:

    def __init__(self, env):

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

        # Set title to name of environment
        self.ax.set_title(env.unwrapped.spec.id)

        # Set axis limits
        self.ax.set_xlim((-100, 100))
        self.ax.set_ylim((-100, 100))
        self.ax.set_zlim((0, 100))

        # Create a representation of the copter
        self.copter = _Vehicle(self.ax)

    def start(self):

        # Instantiate the animator
        anim = animation.FuncAnimation(self.fig, self._animate, interval=int(1000*self.env.dt), blit=False)
        self.fig.canvas.mpl_connect('close_event', self._handle_close)

        # Show the display window
        try:
            plt.show()
        except:
            pass

    def _handle_close(self, event):

        self.is_open = False
        
    def _animate(self, _):

        # Get vehicle position
        x,y,z = self.env.state[0:6:2]

        # Negate Z to accomodate NED
        z = -z

        # Update the copter animation
        self.copter.update(x, y, z)

        # Draw everything
        self.fig.canvas.draw()
