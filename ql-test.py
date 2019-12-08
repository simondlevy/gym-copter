#!/usr/bin/env python3
'''
Tests a copter for maximum altitude using

Copyright (C) Simon D. Levy 2019

MIT License
'''

import pickle
import numpy as np

GAME = 'Copter-v0'

filename = GAME + '.pkl'

print('Loading ' + filename)

with open(filename, 'rb') as f:

    agent = pickle.load(f)

    np.set_printoptions(precision=2)
    print(agent)

    agent.play()
