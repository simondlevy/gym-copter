#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e.ch19 import a2c

args = a2c.parse_args()

test_env = gym.make(args.env)

a2c.train(test_env, args)
