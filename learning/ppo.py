#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e.ch19 import ppo

args = ppo.parse_args()

test_env = gym.make(args.env)

ppo.train(test_env, args)
