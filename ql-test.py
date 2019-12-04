#!/usr/bin/env python3
'''
Tests a quadcopter trained by Q-Learning

Copyright (C) Simon D. Levy 2019

MIT License
'''

from ql import QLAgent
import gym_copter
import gym
import pickle

from sys import stdout
import numpy as np

GAME = 'Copter-v0'

filename = GAME + '.pkl'

with open(filename, 'rb') as f:

    agent = pickle.load(f)

    agent.play()
