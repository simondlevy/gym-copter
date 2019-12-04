#!/usr/bin/env python3
'''
Tests a quadcopter trained by Q-Learning

Copyright (C) Simon D. Levy 2019

MIT License
'''

from ql import QLAgent
import gym_copter
import gym

EPISODES = 5000
ALPHA    = .001
GAMMA    = 0.99
EPSILON  = 0.99
GAME     = 'Copter-v0'

env = gym.make(GAME)
