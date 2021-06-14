#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from pidcontrollers import DescentPidController
from pidcontrollers import PositionHoldPidController

from main import demo2d


def heuristic(state, pidcontrollers):

    y, dy, z, dz, phi, dphi = state

    poshold_pid, descent_pid = pidcontrollers

    pos_todo = poshold_pid.getDemand(y, dy)

    hover_todo = descent_pid.getDemand(z, dz)

    return hover_todo-pos_todo, hover_todo+pos_todo


def main():

    demo2d('gym_copter:Lander2D-v0', heuristic,
           (PositionHoldPidController(), DescentPidController()))


main()