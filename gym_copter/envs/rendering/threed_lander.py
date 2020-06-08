'''
Visualizer for 3D lander

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.threed import ThreeD, create
import numpy as np

class ThreeDLander(ThreeD):

    def __init__(self, env, radius=2):

        ThreeD.__init__(self, env, lim=20, label='Lander', viewangles=[60,135])

        self.circle = create(self.ax, '-', 'r')
        pts = np.linspace(-np.pi, +np.pi, 1000)
        self.circle_x = radius * np.sin(pts)
        self.circle_y = radius * np.cos(pts)
        self.circle_z = np.zeros(self.circle_x.shape)

    def _animate(self, _):

        ThreeD._animate(self, _)

        self.circle.set_data(self.circle_x, self.circle_y)
        self.circle.set_3d_properties(self.circle_z)
