'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym.envs.registration import register

# 2D lander
register(
    id='Lander-v0',
    entry_point='gym_copter.envs:Lander2D',
    max_episode_steps=400
)

# 3D lander without ground target
register(
    id='Lander3D-v0',
    entry_point='gym_copter.envs:Lander3D',
    max_episode_steps=400
)

# 3D lander with X,Y ground target
register(
    id='Lander3D-v1',
    entry_point='gym_copter.envs:Lander3DPoint',
    max_episode_steps=400
)

# 3D lander with discrete inside-of-circle ground target
register(
    id='Lander3D-v2',
    entry_point='gym_copter.envs:Lander3DRing',
    max_episode_steps=400
)
