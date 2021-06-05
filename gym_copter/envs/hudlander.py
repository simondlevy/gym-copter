#!/usr/bin/env python3
'''
HUD demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.hover3d import Hover3D
from gym_copter.envs.lander3d import Lander3D
from gym_copter.rendering.hud import HUD

def main():

    env = Lander3D()

    HUD(env)

    env.demo_heuristic() 

if __name__ == '__main__':

    main()
