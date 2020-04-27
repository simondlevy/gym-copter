'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='CopterTakeoff-v0',
    entry_point='gym_copter.envs:CopterTakeoff',
    max_episode_steps=10000
)

register(
    id='CopterDistance-v0',
    entry_point='gym_copter.envs:CopterDistance',
    max_episode_steps=100000
)

register(
    id='CopterAltHold-v0',
    entry_point='gym_copter.envs:CopterAltHold',
    max_episode_steps=10000
)

register(
    id='CopterTarget-v0',
    entry_point='gym_copter.envs:CopterTarget',
    max_episode_steps=10000
)
