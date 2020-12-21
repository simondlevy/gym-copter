'''
3D Copter-Lander with full dynamics (12 state values)

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class Lander3D(gym.Env, EzPickle):

    # Parameters to adjust  
    INITIAL_ALTITUDE           = 5
    INITIAL_RANDOM_OFFSET      = 2.5
    XY_PENALTY_FACTOR          = 25   # designed so that maximal penalty is around 100
    PITCH_ROLL_PENALTY_FACTOR  = 0 #250   
    YAW_PENALTY_FACTOR         = 50   
    BOUNDS                     = 10
    OUT_OF_BOUNDS_PENALTY      = 100
    RESTING_DURATION           = 1.0  # for rendering for a short while after successful landing
    FRAMES_PER_SECOND          = 50
    MAX_ANGLE                  = 45   # big penalty if roll or pitch angles go beyond this
    EXCESS_ANGLE_PENALTY       = 100

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()

        self.prev_reward = None

        # Observation is all state values
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

        # Action is four floats (one per motor)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

        # Support for rendering
        self.renderer = None
        self.pose = None

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics with random perturbations
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X]      =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Y]      =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Z]      = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        # Stop motors after safe landing
        if status == d.STATUS_LANDED:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            d.setMotors(np.clip(action, 0, 1))    # keep motors in interval [0,1]
            d.update()

        # Get new state from dynamics
        state = np.array(d.getState())

        # Extract pose from state
        x, y, z, phi, theta, psi = state[0::2]

        # Set lander pose in display
        self.pose = x, y, z, phi, theta, psi

        # Reward is a simple penalty for overall distance and angle and their first derivatives
        shaping = -(
                self.XY_PENALTY_FACTOR * np.sqrt(np.sum(state[0:6]**2)) + 
                self.PITCH_ROLL_PENALTY_FACTOR * np.sqrt(np.sum(state[6:10]**2)) +
                self.YAW_PENALTY_FACTOR * np.sqrt(np.sum(state[10:12]**2))
                )
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0
        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go out of bounds
        if abs(x) >= self.BOUNDS or abs(y) >= self.BOUNDS:
            done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        # Lose bigly for excess roll or pitch 
        #if abs(phi) >= self.max_angle or abs(theta) >= self.max_angle:
        #    done = True
        #    reward = -self.OUT_OF_BOUNDS_PENALTY

        # It's all over once we're on the ground
        if status == d.STATUS_LANDED:

            done = True

            # 3D subclass adds a bonus for landing within radius
            reward += self._get_bonus(x, y)

        elif status == d.STATUS_CRASHED:

            # Crashed!
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        '''
        Returns None because we run renderer on a separate thread
        '''

        return None

    def close(self):

        return

    def _get_bonus(self, x, y):

        return 0

## End of Lander3D classes ----------------------------------------------------------------

def heuristic_lander(env, heuristic, renderer=None, seed=None):

    import time

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    while True:

        action = heuristic(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if steps % 20 == 0 or done:
           print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1

        if done: break

        if not renderer is None:
            time.sleep(1./env.FRAMES_PER_SECOND)

    env.close()
    return total_reward
