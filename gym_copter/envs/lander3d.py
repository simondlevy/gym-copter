#!/usr/bin/env python3
'''
3D Copter-Lander class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import time, sleep
import numpy as np

from gym_copter.envs.lander import _Lander
from gym_copter.rendering.hud import HUD
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


class Lander3D(_Lander):

    def __init__(self, obs_size=10):

        _Lander.__init__(self, obs_size, 4)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta']

        self.prev = None

        self.viewer = HUD(self)

    def reset(self):

        return _Lander._reset(self)

    def render(self, mode='human'):

        print(str(self) + ' render')

        if self.prev is not None:
            dt = 1/self.FRAMES_PER_SECOND - 3.0 * (time()-self.prev)
            if dt > 0:
                sleep(dt)

        self.prev = time()

        return self.viewer.render(mode)

    def demo_pose(self, args):

        x, y, z, phi, theta, viewer = args

        while viewer.is_open():

            self._reset(pose=(x, y, z, phi, theta), perturb=False)

            self.render()

            sleep(.01)

        self.close()

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state[:10]


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
