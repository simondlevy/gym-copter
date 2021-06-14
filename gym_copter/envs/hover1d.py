'''
2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.hover import _Hover


class Hover1D(_Hover):

    def __init__(self):

        _Hover.__init__(self, 2, 1)

        # For generating CSV file
        self.STATE_NAMES = ['Z', 'dZ']

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return _Hover._reset(self)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDHoverRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDHoverRenderer(self)

        return self.viewer.render(mode, self.pose, self.spinning)
        # return self.viewer._complete(mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_state(self, state):

        return state[4:6]

    def _get_motors(self, motors):

        return [motors[0], motors[0], motors[0], motors[0]]
