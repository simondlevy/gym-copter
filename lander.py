#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.rendering import ThreeDLanderRenderer

from time import sleep

import numpy as np

import gymnasium as gym

from gym_copter.cmdline import (make_parser, make_parser_3d,
                                wrap, parse_view_angles)
def _constrain(val, lim):
    return -lim if val < -lim else (+lim if val > +lim else val)


class AltitudeHoldPidController:

    def __init__(self, k_p=0.2, k_i=3, k_tgt=5, k_windup=0.2):

        self.k_p = k_p
        self.k_i = k_i

        self.k_tgt = k_tgt

        # Prevents integral windup
        self.k_windup = k_windup

        # Error integral
        self.ei = 0

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (self.k_tgt - z) - dz

        # Compute I term
        self.ei += e

        # avoid integral windup
        self.ei = _constrain(self.ei, self.k_windup)

        return e * self.k_p + self.ei * self.k_i


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.Kd = Kd

        # Accumulated values
        self.lastError = 0

    def getDemand(self, x, dx):

        error = -x - dx

        dterm = (error - self.lastError) * self.Kd

        self.lastError = error

        return dterm


class DescentPidController:

    def __init__(self, k_p=1.15, Kd=1.33):

        self.k_p = k_p
        self.Kd = Kd

    def getDemand(self, z, dz):

        return z*self.k_p + dz*self.Kd


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity

# Threaded
def _demo_heuristic(env, fun, pidcontrollers,
                    seed=None, csvfilename=None, nopid=False):

    env.unwrapped.seed = seed
    np.random.seed(seed)

    total_reward = 0
    steps = 0
    state, _ = env.reset()

    dt = 1. / env.unwrapped.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while True:

        action = np.zeros(actsize) if nopid else fun(state, pidcontrollers)

        state, reward, done, _, _ = env.step(action)

        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.unwrapped.FRAMES_PER_SECOND)
 
        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    env.close()

    if csvfile is not None:
        csvfile.close()



def demo3d(envname, heuristic, pidcontrollers, renderer):

    env = gym.make(envname)

    parser = make_parser_3d()

    args = parser.parse_args()

    viewer = renderer(env,
                      _demo_heuristic,
                      (heuristic, pidcontrollers,
                      args.seed, args.csvfilename, args.nopid),
                      viewangles=parse_view_angles(args),
                      outfile='movie.mp4' if args.movie else None)

    viewer.start()

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

    demo3d('gym_copter:Lander-v0', heuristic,
           pidcontrollers, ThreeDLanderRenderer)


if __name__ == '__main__':

    main()
