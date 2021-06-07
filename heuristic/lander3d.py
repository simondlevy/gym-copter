#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym

from heuristic import demo_heuristic

from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import DescentPidController

from parsing import parse3d

from gym_copter.rendering.threed import ThreeDLanderRenderer


def heuristic(env, state, pidcontrollers):
    '''
    PID controller
    '''
    (phi_rate_pid,
     theta_rate_pid,
     x_poshold_pid,
     y_poshold_pid,
     descent_pid) = pidcontrollers

    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state

    phi_todo = 0
    theta_todo = 0

    phi_rate_todo = phi_rate_pid.getDemand(dphi)
    y_pos_todo = x_poshold_pid.getDemand(y, dy)
    phi_todo = phi_rate_todo + y_pos_todo

    theta_rate_todo = theta_rate_pid.getDemand(-dtheta)
    x_pos_todo = y_poshold_pid.getDemand(x, dx)
    theta_todo = theta_rate_todo + x_pos_todo

    descent_todo = descent_pid.getDemand(z, dz)

    t, r, p = (descent_todo+1)/2, phi_todo, theta_todo

    # Use mixer to set motors
    return t-r-p, t+r+p, t+r-p, t-r+p


def main():

    args = parse3d()

    env = gym.make('gym_copter:Lander3D-v0')

    pidcontrollers = (
                      AngularVelocityPidController(),
                      AngularVelocityPidController(),
                      PositionHoldPidController(),
                      PositionHoldPidController(),
                      DescentPidController()
                     )

    if args.hud:

        env.use_hud()

        demo_heuristic(env, args.seed, args.csvfilename)

    else:

        viewangles = tuple((int(s) for s in args.view.split(',')))

        viewer = ThreeDLanderRenderer(env,
                                      demo_heuristic,
                                      (heuristic, pidcontrollers,
                                       args.seed, args.csvfilename),
                                      viewangles=viewangles)

        viewer.start()


if __name__ == '__main__':

    main()
