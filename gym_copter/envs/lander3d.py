#!/usr/bin/env python3
'''
3D Copter-Lander class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.envs.threed import _ThreeD
from gym_copter.envs.lander import _Lander
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


class Lander3D(_Lander, _ThreeD):

    def __init__(self, obs_size=10):

        _Lander.__init__(self, obs_size, 4)
        _ThreeD.__init__(self)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta']

        self.viewer = None

    def reset(self):

        return _Lander._reset(self)

    def _get_state(self, state):

        return state[:10]

    def use_hud(self):

        _ThreeD.use_hud(self)

    def render(self, mode='human'):

        return _ThreeD.render(self, mode)

    def demo_pose(self, args):

        _ThreeD.demo_pose(self, args)


class LanderVisual(Lander3D):

    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):

        Lander3D.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):

        result = Lander3D.step(self, action)

        x, y, z, phi, theta, psi = self.pose

        self.image = self.vs.getImage(x,
                                      y,
                                      max(-z, 1e-6),  # keep Z positive
                                      np.degrees(phi),
                                      np.degrees(theta),
                                      np.degrees(psi))

        return result

    def render(self, mode='human'):

        if self.image is not None:
            self.vs.display_image(self.image)


class LanderDVS(LanderVisual):

    def __init__(self):

        LanderVisual.__init__(self, vs=DVS(res=LanderVisual.RES))
