#!/usr/bin/env python3
'''
Simple test of gym-copter

Copyright (C) Simon D. Levy 2019

MIT License
'''

import gym
from time import sleep

env = gym.make('gym_copter:copter-v0')
env.reset()

while True:
    env.render()
    state, _, _, _ = env.step([.6]*4)
    print(state.pose.location[2])
env.close()
