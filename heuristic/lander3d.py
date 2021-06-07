#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym

from time import sleep
import numpy as np

from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import DescentPidController

from parsing import make_parser

from gym_copter.rendering.threed import ThreeDLanderRenderer


def heuristic(env,
              state,
              phi_rate_pid,
              x_poshold_pid,
              theta_rate_pid,
              y_poshold_pid,
              descent_pid):
    '''
    PID controller
    '''
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


def demo_heuristic(env, seed=None, csvfilename=None):
    '''
    csvfile arg will only be added by 3D scripts.
    '''

    phi_rate_pid = AngularVelocityPidController()
    theta_rate_pid = AngularVelocityPidController()
    x_poshold_pid = PositionHoldPidController()
    y_poshold_pid = PositionHoldPidController()
    descent_pid = DescentPidController()

    env.seed(seed)
    np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    dt = 1. / env.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while True:

        action = heuristic(env,
                           state,
                           phi_rate_pid,
                           x_poshold_pid,
                           theta_rate_pid,
                           y_poshold_pid,
                           descent_pid)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    sleep(1)
    env.close()
    print('loop: ' + str(env) + ' ' + str(env.viewer))
    if csvfile is not None:
        csvfile.close()
    return total_reward


def main():

    parser = make_parser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--hud', action='store_true',
                       help='Use heads-up display')

    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')

    args = parser.parse_args()

    env = gym.make('gym_copter:Lander3D-v0')

    if args.hud:

        env.use_hud()

        demo_heuristic(env, args.seed, args.csvfilename)

    else:

        viewangles = tuple((int(s) for s in args.view.split(',')))

        viewer = ThreeDLanderRenderer(env,
                                      demo_heuristic,
                                      (env, args.seed, args.csvfilename),
                                      viewangles=viewangles)

        viewer.start()


if __name__ == '__main__':

    main()
