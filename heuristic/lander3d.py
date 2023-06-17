#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from main import demo3d

from pidcontrollers import PositionHoldPidController, DescentPidController

from gym_copter.rendering.threed import ThreeDLanderRenderer


def heuristic(state, pidcontrollers):
    '''
    PID controller
    '''

    x_poshold_pid, y_poshold_pid, descent_pid = pidcontrollers

    x, dx, y, dy, z, dz, _, _, _, _ = state

    y_pos_todo = x_poshold_pid.getDemand(y, dy)

    x_pos_todo = y_poshold_pid.getDemand(x, dx)

    descent_todo = descent_pid.getDemand(z, dz)

    t, r, p = (descent_todo+1)/2, y_pos_todo, x_pos_todo

    # Use mixer to set motors
    return t-r-p, t+r+p, t+r-p, t-r+p


def main():

    pidcontrollers = (PositionHoldPidController(),
                      PositionHoldPidController(),
                      DescentPidController())

    demo3d('gym_copter:Lander3D-v0', heuristic,
           pidcontrollers, ThreeDLanderRenderer)


if __name__ == '__main__':

    main()
