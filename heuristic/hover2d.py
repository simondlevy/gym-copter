#!/usr/bin/env python3
'''
Heuristic demo for 2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from pidcontrollers import AltitudeHoldPidController
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController

from main import demo


def heuristic(state, pidcontrollers):

    y, dy, z, dz, phi, dphi = state

    rate_pid, pos_pid, alt_pid = pidcontrollers

    rate_todo = rate_pid.getDemand(dphi)
    pos_todo = pos_pid.getDemand(y, dy)

    phi_todo = rate_todo + pos_todo

    hover_todo = alt_pid.getDemand(z, dz)

    return hover_todo-phi_todo, hover_todo+phi_todo


demo('gym_copter:Hover2D-v0', heuristic,
     (AngularVelocityPidController(),
      PositionHoldPidController(),
      AltitudeHoldPidController()))
