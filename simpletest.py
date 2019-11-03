#!/usr/bin/env python3
'''
Simple test of gym-copter

Copyright (C) Simon D. Levy 2019

MIT License
'''

import gym

env = gym.make('gym_copter:copter-v0')

env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample() # take a random action
    print(action)
    env.step(action)

env.close()
