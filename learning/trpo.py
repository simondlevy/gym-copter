#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e_ch19 import trpo

args = trpo.parse_args()

test_env = gym.make(args.env)

trpo.train(test_env, args)
