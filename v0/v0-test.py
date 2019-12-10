#!/usr/bin/env python3
'''
Tests a copter using a simple pre-computed Q-table that favors fully-on motors

Copyright (C) Simon D. Levy 2019

MIT License
'''

import numpy as np
from ql import QLAgent
import gym
import gym_copter

env = gym.make('Copter-v0')
q_table = np.array([[0,1]])
agent = QLAgent(env, q_table)
agent.play()
