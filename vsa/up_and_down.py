#!/usr/bin/env python3
'''
Heuristic demo for 1D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import gym


def _constrain(val, lim):
    return -lim if val < -lim else (+lim if val > +lim else val)


def main():

    ALTITUDE_TARGETS = 1,3,5
    DURATION = 10
    ALTITUDE_START = 3

    K_P = 0.2
    K_I = 3
    K_WINDUP = 0.2

    # Error integral
    ei = 0

    # Start CSV file
    filename = (
        'ud_k_start=%2.2f_kp=%2.2f_Ki=%2.2f_k_windup=%2.2f.csv' %
        (ALTITUDE_START, K_P, K_I, K_WINDUP))
    csvfile = open(filename, 'w')
    csvfile.write('t,z,dz,e,ei,u\n')

    env = gym.make('gym_copter:Hover1D-v0')

    env.set_altitude(ALTITUDE_START)

    target_index = 0
    total_reward = 0
    steps = 0
    state = env.reset()

    total_steps = DURATION * env.FRAMES_PER_SECOND

    while steps < total_steps

        t = steps / env.FRAMES_PER_SECOND

        z, dz = state

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (ALTITUDE_TARGETS[target_index] - z) - dz

        # Compute I term
        ei += e

        # Avoid integral windup
        ei = _constrain(ei, K_WINDUP)

        # Compute demand u
        u = e * K_P + ei * K_I

        # Write current values to CSV file
        csvfile.write('%3.3f,%3.3f,%3.3f,%3.3f,%3.3f,%3.3f\n' %
                      (t, z, dz, e, ei, u))

        state, reward, done, _ = env.step((u,))

        total_reward += reward

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    env.close()


main()
