'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='copter-v0',
    entry_point='gym_copter.envs:CopterEnv',
)
register(
    id='copter-extrahard-v0',
    entry_point='gym_copter.envs:CopterExtraHardEnv',
)
