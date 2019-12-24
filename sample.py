#!/usr/bin/env python3
'''
Run classic random-sampling demo on gym-copter

Copyright (C) 2019 Simon D. Levy

MIT License
'''


import gym
import gym_copter

env = gym.make('Copter-v2')

observation = env.reset()

for _ in range(1000):
  env.render()
  action = env.action_space.sample() 
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()

env.close()
