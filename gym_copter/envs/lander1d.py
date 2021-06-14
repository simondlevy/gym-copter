'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander import _Lander


class Lander1D(_Lander):

    def __init__(self):

        _Lander.__init__(self, 2, 1)

        # For generating CSV file
        self.STATE_NAMES = ['Z', 'dZ']

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

        return state[4:6]

    def _get_motors(self, motors):

        return [motors[0], motors[0], motors[0], motors[0]]
