'''
Support for 3D copter environments

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep

from gym_copter.rendering.hud import HUD


class _ThreeD:

    def use_hud(self):

        self.viewer = HUD(self)

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
