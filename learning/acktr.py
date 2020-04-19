#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e_ch19 import acktr

args = acktr.parse_args()

test_env = gym.make(args.env)

acktr.train(test_env, args)
