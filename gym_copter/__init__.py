'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gymnasium.envs.registration import register

# 1D lander
register(
    id='Lander1D-v0',
    entry_point='gym_copter.envs:Lander1D',
    max_episode_steps=400
)

# 1D hover
register(
    id='Hover1D-v0',
    entry_point='gym_copter.envs:Hover1D',
    max_episode_steps=1000
)

# 2D lander
register(
    id='Lander2D-v0',
    entry_point='gym_copter.envs:Lander2D',
    max_episode_steps=400
)

# 2D hover
register(
    id='Hover2D-v0',
    entry_point='gym_copter.envs:Hover2D',
    max_episode_steps=1000
)

# 3D lander
register(
    id='Lander-v0',
    entry_point='gym_copter.envs:Lander',
    max_episode_steps=400
)

# 3D hover
register(
    id='Hover3D-v0',
    entry_point='gym_copter.envs:Hover3D',
    max_episode_steps=1000
)
