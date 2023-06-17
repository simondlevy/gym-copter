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

    ALTITUDE_TARGETS = 1, 3, 5
    DURATION = 10
    ALTITUDE_START = 3

    K_P = 0.2
    K_NEUTRAL = 0.524
    K_WINDUP = 0.2

    # Start CSV file
    filename = (
        'targets=%d_%d_%d_start=%d_kp=%2.2f_Kneut=%2.2f_k_windup=%2.2f.csv' %
        (*ALTITUDE_TARGETS, ALTITUDE_START, K_P, K_NEUTRAL, K_WINDUP))
    csvfile = open(filename, 'w')
    csvfile.write('time,target,z,dz,e,u\n')

    env = gym.make('gym_copter:Hover1D-v0')

    env.set_altitude(ALTITUDE_START)

    target_index = 0
    total_reward = 0
    state = env.reset()

    total_steps = DURATION * env.FRAMES_PER_SECOND

    steps_per_altitude = int(total_steps / len(ALTITUDE_TARGETS))

    for step in range(total_steps):

        t = step / env.FRAMES_PER_SECOND

        z, dz = state

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Support changing targets periodically
        target = ALTITUDE_TARGETS[target_index]

        # Compute error as scaled target minus actual
        e = (target - z) - dz

        # Compute demand u
        u = e * K_P + K_NEUTRAL

        # Write current values to CSV file
        csvfile.write('%3.3f,%3.3f,%3.3f,%3.3f,%3.3f,%3.3f\n' %
                      (t, target, z, dz, e, u))

        state, reward, done, _ = env.step((u,))

        total_reward += reward

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        if (step % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (step, total_reward))

        if (step > 0) and (step % steps_per_altitude == 0):
            target_index += 1

        if done:
            break

    env.close()


main()
