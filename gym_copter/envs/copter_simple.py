'''
gym-copter Environment class with simplified physics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.copter_env import CopterEnv
from gym import spaces
import numpy as np

class CopterSimple(CopterEnv):
    '''
    A simplified copter class for Q-Learning.
    Action space (motor values) and observation space (altitude) are discretized.
    '''

    ALTITUDE_MAX = 10
    MOTOR_STEPS  = 5

    def __init__(self):

        CopterEnv.__init__(self)

        # Action space = motors, discretize to intervals
        self.action_space = spaces.Discrete((CopterEnvAltitudeRewardDiscreteMotors.MOTOR_STEPS+1)**4)

        # Observation space = altitude
        self.observation_space = spaces.Box(np.array([0]), np.array([100]))

    def step(self, action):

        # Convert discrete action index to array of floating-point number values
        motors = [(action//(self.MOTOR_STEPS+1)**k)%(self.MOTOR_STEPS+1)/float(self.MOTOR_STEPS) for k in range(4)]

        # Call parent-class step() to do basic update
        state, reward, episode_over, info = CopterEnv.step(self, motors)

        # Maximum altitude attained: set episode-over flag
        if self.altitude > self.ALTITUDE_MAX:
            episode_over = True 

        # Altitude is both the state and the reward
        return np.array([self.altitude]), self.altitude, episode_over, info

    def reset(self):
        CopterEnv.reset(self)
        self.airborne = False
        return np.array([self.altitude])

