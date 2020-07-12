'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

register(
    id='Lander-v2',
    entry_point='gym_copter.envs:Lander2D',
    max_episode_steps=10000
)

register(
    id='Lander-v3',
    entry_point='gym_copter.envs:Lander3D',
    max_episode_steps=10000
)

register(
    id='Distance-v0',
    entry_point='gym_copter.envs:Distance',
    max_episode_steps=1000
)

register(
    id='Takeoff-v0',
    entry_point='gym_copter.envs:Takeoff',
    max_episode_steps=1000
)
