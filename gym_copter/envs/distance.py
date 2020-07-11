#!/usr/bin/env python3
'''
3D distance-maximizing environment

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class Distance(gym.Env, EzPickle):

    FRAMES_PER_SECOND = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()

        self.prev_reward = None

        # Observation is all state values except yaw and its derivative
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        # Action is motor values
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

        # Support for rendering
        self.renderer = None
        self.pose = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics
        state = np.zeros(12)
        self.dynamics.setState(state)

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        d.setMotors(action)
        d.update()

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta, psi, _ = d.getState()

        # Set pose in display
        self.pose = posx, posy, posz, phi, theta, psi

        # Convert state to usable form
        state = np.array([posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta])

        reward = 0

        # Assume we're not done yet
        done = False

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        from gym_copter.rendering.threed import ThreeDDistanceRenderer

        # Create renderer if not done yet
        if self.renderer is None:
            self.renderer = ThreeDDistanceRenderer(self, self.LANDING_RADIUS)

        return self.renderer.render()

    def close(self):

        return

## End of Distance class ----------------------------------------------------------------

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the X coordinate
                  s[1] is the X speed
                  s[2] is the Y coordinate
                  s[3] is the Y speed
                  s[4] is the vertical coordinate
                  s[5] is the vertical speed
                  s[6] is the roll angle
                  s[7] is the roll angular speed
                  s[8] is the pitch angle
                  s[9] is the pitch angular speed
     returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = s

    return np.ones(4)

def heuristic_distance(env, renderer=None, seed=None):

    import time

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    while True:

        action = heuristic(env,state)
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


if __name__ == '__main__':

    from gym_copter.rendering.threed import ThreeDDistanceRenderer
    import threading

    env = Distance()

    renderer = ThreeDDistanceRenderer(env)

    thread = threading.Thread(target=heuristic_distance, args=(env, renderer))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start()    
