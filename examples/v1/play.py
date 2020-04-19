#!/usr/bin/env python3

import gym

import gym_copter

from drlho2e_ch19 import play

args = play.parse_args()

test_env = gym.make(args.env)

play.play(test_env, args)
