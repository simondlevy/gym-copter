'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Copter-v0',
    entry_point='gym_copter.envs:CopterTakeoff',
    max_episode_steps=10000
)

register(
    id='Copter-v1',
    entry_point='gym_copter.envs:CopterDistance',
    max_episode_steps=10000
)

register(
    id='Copter-v2',
    entry_point='gym_copter.envs:CopterAltHold',
    max_episode_steps=10000
)
