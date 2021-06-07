'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.parsing import _make_parser
from gym_copter.envs.lander import _Lander
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController


class Lander2D(_Lander):

    def __init__(self):

        _Lander.__init__(self, 6, 2)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Z', 'dZ', 'Phi', 'dPhi']

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return _Lander._reset(self)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDLanderRenderer(self)

        return self.viewer.render(mode, self.pose, self.spinning)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_state(self, state):

        return state[2:8]

    def _get_motors(self, motors):

        return [motors[0], motors[1], motors[1], motors[0]]
