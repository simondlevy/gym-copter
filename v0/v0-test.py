#!/usr/bin/env python3
'''
Tests a copter for maximum altitude using

Copyright (C) Simon D. Levy 2019

MIT License
'''

import numpy as np
from ql import QLAgent
import gym
import gym_copter
import pickle
from sys import argv

def load_agent():
    f = open('Copter-v0.pkl', 'rb')
    agent = pickle.load(f)
    f.close()
    return agent

def make_agent():
    env = gym.make('Copter-v0')
    q_table = np.array([[0,1]])
    return QLAgent(env, q_table)

agent = load_agent() if len(argv) < 1 else make_agent()

agent.play()
