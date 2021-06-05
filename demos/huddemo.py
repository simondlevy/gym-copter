#!/usr/bin/env python3
'''
HUD demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym
from gym import wrappers

from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController
from gym_copter.pidcontrollers import DescentPidController


def main():

    phi_rate_pid = AngularVelocityPidController()
    theta_rate_pid = AngularVelocityPidController()
    x_poshold_pid = PositionHoldPidController()
    y_poshold_pid = PositionHoldPidController()
    descent_pid = DescentPidController()

    env = gym.make('gym_copter:Lander3D-v0')

    env = wrappers.Monitor(env, 'movie/', force=True)

    state = env.reset()

    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state

    while True:

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
        action = t-r-p, t+r+p, t+r-p, t-r+p

        state, _, done, _ = env.step(action)

        if done:
            break

        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state

    env.close()


if __name__ == '__main__':

    main()
