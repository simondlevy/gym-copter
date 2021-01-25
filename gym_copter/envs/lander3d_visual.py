#!/usr/bin/env python3
'''
3D Copter-Lander class with visual target

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import Lander3D, demo


class Lander3DVisual(Lander3D):

    # 3D model
    OBSERVATION_SIZE = 10
    ACTION_SIZE = 4

    def __init__(self):

        Lander3D.__init__(self, view_width=0.5)


if __name__ == '__main__':

    demo(Lander3DVisual())
