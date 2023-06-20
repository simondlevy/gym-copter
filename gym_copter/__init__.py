'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gymnasium.envs.registration import register

register(
    id='Hover3D-v0',
    entry_point='gym_copter.envs:Hover3D',
    max_episode_steps=1000
)
