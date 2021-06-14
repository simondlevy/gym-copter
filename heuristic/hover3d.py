#!/usr/bin/env python3
'''
3D Copter-Hover heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from main import demo3d

from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import AltitudeHoldPidController

from gym_copter.rendering.threed import ThreeDHoverRenderer


def heuristic(state, pidcontrollers):
    '''
    PID controller
    '''
    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, _, dpsi = state

    (roll_rate_pid,
     pitch_rate_pid,
     yaw_rate_pid,
     x_poshold_pid,
     y_poshold_pid,
     althold_pid) = pidcontrollers

    roll_rate_todo = roll_rate_pid.getDemand(dphi)
    y_pos_todo = x_poshold_pid.getDemand(y, dy)

    pitch_rate_todo = pitch_rate_pid.getDemand(-dtheta)
    x_pos_todo = y_poshold_pid.getDemand(x, dx)

    roll_todo = roll_rate_todo + y_pos_todo
    pitch_todo = pitch_rate_todo + x_pos_todo
    yaw_todo = yaw_rate_pid.getDemand(-dpsi)

    hover_todo = althold_pid.getDemand(z, dz)

    t, r, p, y = (hover_todo+1)/2, roll_todo, pitch_todo, yaw_todo

    # Use mixer to set motors
    return t-r-p-y, t+r+p-y, t+r-p+y, t-r+p+y


def main():

    pidcontrollers = (
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      PositionHoldPidController(),
                      PositionHoldPidController(),
                      AltitudeHoldPidController()
                     )

    demo3d('gym_copter:Hover3D-v0', heuristic,
           pidcontrollers, ThreeDHoverRenderer)


if __name__ == '__main__':

    main()
