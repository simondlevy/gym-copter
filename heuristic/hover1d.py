#!/usr/bin/env python3
'''
Heuristic demo for 2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from pidcontrollers import AltitudeHoldPidController

from main import demo


def heuristic(state, pidcontrollers):

    z, dz = state

    alt_pid = pidcontrollers[0]

    return (alt_pid.getDemand(z, dz),)


demo('gym_copter:Hover1D-v0', heuristic, (AltitudeHoldPidController(),))
