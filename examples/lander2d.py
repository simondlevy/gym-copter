#!/usr/bin/env python3

import numpy as np
import gym
import gym_copter

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    throttle_targ = 0.55 * np.abs(s[0])           # target y should be proportional to horizontal offset

    throttle_todo = (throttle_targ - s[1])*2 - s[3]*8

    throttle_todo = throttle_todo*10 - 1

    throttle_todo = np.clip(throttle_todo, -1, +1)

    roll_todo = s[0]/100

    return np.array([throttle_todo, roll_todo])


if __name__ == '__main__':

    env = gym.make('CopterLander-v1')
    total_reward = 0
    steps = 0
    s = env.reset(xoff=4)
    while True:
        a = heuristic(env, s)
        s, r, done, _ = env.step(a)
        total_reward += r

        still_open = env.render()
        if still_open == False: break
        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    env.close()


