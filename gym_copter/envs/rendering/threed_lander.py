'''
Visualizer for 3D lander

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.threed import ThreeD, _Vehicle

class ThreeDLander(ThreeD):

    def __init__(self, env, title):

        ThreeD.__init__(self, env, title)

        self.circle = _Vehicle._create(self.ax, '-', 'r')

    def _animate(self, _):

       ThreeD._animate(self, _)

       self.circle.set_data([0, 10], [0, 10])
       self.circle.set_3d_properties([0,0])
