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
    TICKS_MAX    = 20
    MOTOR_STEPS  = 5

    def __init__(self):

        CopterEnv.__init__(self)

        # Action space = motors, discretize to intervals
        self.action_space = spaces.Discrete(self.MOTOR_STEPS+1)

        # Observation space = altitude, discretized to meters
        self.observation_space = spaces.Discrete(self.ALTITUDE_MAX+1)

    def step(self, action):

        # Convert discrete action index to array of floating-point number values
        #motors = [(action//(self.MOTOR_STEPS+1)**k)%(self.MOTOR_STEPS+1)/float(self.MOTOR_STEPS) for k in range(4)]
        motors = [action / float(self.MOTOR_STEPS) for _ in range(4)]

        motors = [0,0,0,0]
        print(self.state)

        # Call parent-class step() to do basic update
        state, reward, episode_over, info = CopterEnv.step(self, motors)

        print(state)

        exit(0)

        # Dynamics uses NED coordinates, so negate to get altitude; then cap at max
        altitude = min(int(-state[5]), self.ALTITUDE_MAX)

        print('motors=', motors, '    altitude = ', altitude)

        # Maximum altitude attained: set episode-over flag
        if altitude > (self.ALTITUDE_MAX-1):
            episode_over = True 

        # Too many ticks elapsed: set episode-over flag
        if self.ticks > self.TICKS_MAX:
            episode_over = True

        # Altitude is both the state and the reward
        return np.array([altitude]), altitude, episode_over, info

    def reset(self):
        CopterEnv.reset(self)
        return np.array([0])

