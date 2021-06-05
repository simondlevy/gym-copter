#!/usr/bin/env python3
'''
HUD demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import Lander3D
from gym_copter.rendering.hud import HUD

def main():

    env = Lander3D()

    hud = HUD()

if __name__ == '__main__':

    main()
