#!/usr/bin/env python3
'''
Trains a copter for maximum altitude using Q-Learing algorithm

Copyright (C) Simon D. Levy 2019

MIT License
'''

from ql import QLAgent
import gym_copter
import gym
import matplotlib.pyplot as plt
import pickle

EPISODES = 1000
ALPHA    = .1
GAMMA    = 0.5
EPSILON  = 0.5
GAME     = 'Copter-v0'

env = gym.make(GAME)

agent = QLAgent(env)

moving_average_rewards = agent.train(EPISODES, ALPHA, GAMMA, EPSILON, report=1)

plt.plot(moving_average_rewards)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.title(GAME)
plt.show()

filename = GAME + '.pkl'

print('Saving ' + filename)

with open(filename, 'wb') as f:
    pickle.dump(agent, f)
