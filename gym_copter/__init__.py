'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Copter-v0',
    entry_point='gym_copter.envs:CopterSimple',
)

register(
    id='Copter-v1',
    entry_point='gym_copter.envs:CopterEnvRealistic',
)
