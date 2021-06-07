#!/usr/bin/env python3
'''
Heuristic demo for 2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym
from gym import wrappers

from heuristic import demo_heuristic

from parsing import make_parser
from pidcontrollers import AltitudeHoldPidController
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController


def heuristic(state, pidcontrollers):

    y, dy, z, dz, phi, dphi = state

    rate_pid, pos_pid, alt_pid = pidcontrollers

    rate_todo = rate_pid.getDemand(dphi)
    pos_todo = pos_pid.getDemand(y, dy)

    phi_todo = rate_todo + pos_todo

    hover_todo = alt_pid.getDemand(z, dz)

    return hover_todo-phi_todo, hover_todo+phi_todo


def main():

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make('gym_copter:Hover2D-v0')

    env = wrappers.Monitor(env, 'movie/', force=True)

    pidcontrollers = (AngularVelocityPidController(),
                      PositionHoldPidController(),
                      AltitudeHoldPidController())

    demo_heuristic(env, heuristic, pidcontrollers,
                   seed=args.seed, csvfilename=args.csvfilename)

    env.close()


if __name__ == '__main__':
    main()
