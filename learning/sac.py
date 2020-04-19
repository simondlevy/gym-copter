#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e.ch19 import sac

args = sac.parse_args()

test_env = gym.make(args.env)

sac.train(test_env, args)
