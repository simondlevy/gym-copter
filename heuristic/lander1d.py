#!/usr/bin/env python3
'''
1D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from pidcontrollers import DescentPidController
from main import demo


def heuristic(state, pidcontrollers):

    z, dz = state

    descent_pid = pidcontrollers[0]

    return (descent_pid.getDemand(z, dz),)


def main():

    demo('gym_copter:Lander1D-v0', heuristic, (DescentPidController(),))


main()
