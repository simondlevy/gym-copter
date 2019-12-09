#!/usr/bin/env python3
'''
Trains a copter for maximum altitude using Q-Learing algorithm

Copyright (C) Simon D. Levy 2019

MIT License
'''

from ql import QLAgent
import gym_copter
import gym
import pickle

EPISODES = 1
ALPHA    = .1
GAMMA    = 0.9
EPSILON  = 0.1
GAME     = 'Copter-v0'

env = gym.make(GAME)

agent = QLAgent(env)

agent.train(EPISODES, ALPHA, GAMMA, EPSILON)

filename = GAME + '.pkl'

print('Saving ' + filename)

with open(filename, 'wb') as f:
    pickle.dump(agent, f)

print(agent.q_table)
