#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from time import time, sleep

import numpy as np
import gym
from gym import wrappers

from heuristic import demo_heuristic

from parsing import make_parser
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import DescentPidController


def heuristic(state, rate_pid, poshold_pid, descent_pid):

    y, dy, z, dz, phi, dphi = state

    phi_todo = 0

    rate_todo = rate_pid.getDemand(dphi)
    pos_todo = poshold_pid.getDemand(y, dy)

    phi_todo = rate_todo + pos_todo

    hover_todo = descent_pid.getDemand(z, dz)

    return hover_todo-phi_todo, hover_todo+phi_todo



def main():

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make('gym_copter:Lander2D-v0')

    env = wrappers.Monitor(env, 'movie/', force=True)

    demo_heuristic(env, seed=args.seed, csvfilename=args.csvfilename)

    env.close()


if __name__ == '__main__':
    main()
