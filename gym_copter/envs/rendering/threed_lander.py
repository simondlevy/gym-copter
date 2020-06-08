'''
Visualizer for 3D lander

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.threed import ThreeD

class ThreeDLander(ThreeD):

    def __init__(self, env, title):

        ThreeD.__init__(self, env, title)

    def _animate(self, _):

       ThreeD._animate(self, _)
