#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import wrappers

from heuristic import demo_heuristic

from parsing import make_parser
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import DescentPidController


def heuristic(env, state, pidcontrollers):

    y, dy, z, dz, phi, dphi = state

    rate_pid, poshold_pid, descent_pid = pidcontrollers

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

    pidcontrollers = (
        AngularVelocityPidController(),
        PositionHoldPidController(),
        DescentPidController(),
    )

    demo_heuristic(env, heuristic, pidcontrollers,
                   seed=args.seed, csvfilename=args.csvfilename)

    env.close()


if __name__ == '__main__':
    main()
