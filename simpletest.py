#!/usr/bin/env python3
'''
Simple test of gym-copter

Copyright (C) Simon D. Levy 2019

MIT License
'''

import gym

env = gym.make('MountainCar-v0')

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

env.close()
