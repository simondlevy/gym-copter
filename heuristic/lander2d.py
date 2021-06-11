#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from pidcontrollers2 import AngularVelocityPidController
from pidcontrollers2 import PositionHoldPidController
from pidcontrollers import DescentPidController

from main import demo2d


def heuristic(state, pidcontrollers):

    y, dy, z, dz, phi, dphi = state

    rate_pid, poshold_pid, descent_pid = pidcontrollers

    phi_todo = 0

    rate_todo = rate_pid.getDemand(dphi)
    pos_todo = poshold_pid.getDemand(dy)

    phi_todo = rate_todo + pos_todo

    hover_todo = descent_pid.getDemand(z, dz)

    return hover_todo-phi_todo, hover_todo+phi_todo


def main():

    demo2d('gym_copter:Lander2D-v0', heuristic,
           (AngularVelocityPidController(),
            PositionHoldPidController(),
            DescentPidController()))


main()
