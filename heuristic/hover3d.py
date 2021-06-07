#!/usr/bin/env python3
'''
3D Copter-Hover heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym

from gym_copter.rendering.threed import ThreeDHoverRenderer

from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import AltitudeHoldPidController

from parsing import make_3d_parser
from heuristic import demo_heuristic


def heuristic(env, state, pidcontrollers):
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

    roll_todo = 0
    pitch_todo = 0
    yaw_todo = 0

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

    parser = make_3d_parser()

    args = parser.parse_args()

    env = gym.make('gym_copter:Hover3D-v0')

    pidcontrollers = (
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      PositionHoldPidController(),
                      PositionHoldPidController(),
                      AltitudeHoldPidController()
                     )

    if args.hud:

        env.use_hud()

        demo_heuristic(env, heuristic, pidcontrollers,
                       args.seed, args.csvfilename)

    else:

        viewangles = tuple((int(s) for s in args.view.split(',')))

        viewer = ThreeDHoverRenderer(env,
                                     demo_heuristic,
                                     (heuristic, pidcontrollers,
                                      args.seed, args.csvfilename),
                                     viewangles=viewangles)

        viewer.start()


if __name__ == '__main__':

    main()
