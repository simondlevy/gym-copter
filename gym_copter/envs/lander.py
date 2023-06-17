#!/usr/bin/env python3
'''
3D Copter-Lander class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.envs.task import _Task
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS


class Lander(_Task):

    TARGET_RADIUS = 2
    YAW_PENALTY_FACTOR = 50
    XYZ_PENALTY_FACTOR = 25
    DZ_MAX = 10
    DZ_PENALTY = 100

    INSIDE_RADIUS_BONUS = 100

    def __init__(self, obs_size=10):

        _Task.__init__(self, obs_size, 4)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta']

        self.viewer = None

    def reset(self):

        return _Task._reset(self)

    def _get_state(self, state):

        keys = ('x', 'dx', 'y', 'dy', 'z', 'dz',
                'phi', 'dphi', 'theta', 'dtheta')

        return [val for val in [state[key] for key in keys]]

    def _get_reward(self, status, state, d, x, y):

        statepos = np.array([state[v] for v in ('x', 'dx', 'y', 'dy', 'z', 'dz')])
        statepsi = np.array([state[v] for v in ('psi', 'dpsi')])

        # Get penalty based on state and motors
        shaping = -(self.XYZ_PENALTY_FACTOR*np.sqrt(np.sum(statepos**2)) +
                    self.YAW_PENALTY_FACTOR*np.sqrt(np.sum(statepsi**2)))

        if (abs(state['dz']) > self.DZ_MAX):
            shaping -= self.DZ_PENALTY

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)

        self.prev_shaping = shaping

        if status == d.STATUS_LANDED:

            self.done = True
            self.spinning = False

            # Win bigly we land safely between the flags
            if np.sqrt(x**2+y**2) < self.TARGET_RADIUS:

                reward += self.INSIDE_RADIUS_BONUS

        return reward
    def render(self, mode='human'):

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

class LanderVisual(Lander):

    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):

        Lander.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):

        result = Lander.step(self, action)

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
