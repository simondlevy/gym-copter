#!/usr/bin/env python3
'''
Trains a copter for maximum altitude using Q-Learing algorithm

Copyright (C) Simon D. Levy 2019

MIT License
'''

from ql import QLAgent
import gym_copter
import gym
from sys import stdout

EPISODES = 1
ALPHA    = .1
GAMMA    = 0.9
EPSILON  = 0.1
GAME     = 'Copter-v0'

env = gym.make(GAME)

agent = QLAgent(env)

agent.train(EPISODES, ALPHA, GAMMA, EPSILON)

print('Q-table:', agent.q_table)
stdout.flush()

agent.play()
