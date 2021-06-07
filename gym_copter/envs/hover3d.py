'''
3D Copter-Hover class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import time, sleep
from numpy import degrees
import threading

from gym_copter.envs.hover import _Hover
from gym_copter.rendering.threed import ThreeDHoverRenderer
from gym_copter.rendering.hud import HUD
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


class Hover3D(_Hover):

    def __init__(self, obs_size=12):

        _Hover.__init__(self, obs_size, 4)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

        self.prev = None

    def reset(self):

        return _Hover._reset(self)

    def use_hud(self):

        self.viewer = HUD(self)

    def render(self, mode='human'):

        if self.prev is not None:
            dt = 1/self.FRAMES_PER_SECOND - 3.0 * (time()-self.prev)
            if dt > 0:
                sleep(dt)

        self.prev = time()

        return None if self.viewer is None else self.viewer.render(mode)

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

        return state


class HoverVisual(Hover3D):

    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):

        Hover3D.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):

        result = Hover3D.step(self, action)

        x, y, z, phi, theta, psi = self.pose

        self.image = self.vs.getImage(x,
                                      y,
                                      max(-z, 1e-6),  # keep Z positive
                                      degrees(phi),
                                      degrees(theta),
                                      degrees(psi))

        return result

    def render(self, mode='human'):

        if self.image is not None:
            self.vs.display_image(self.image)


class HoverDVS(HoverVisual):

    def __init__(self):

        HoverVisual.__init__(self, vs=DVS(res=HoverVisual.RES))
