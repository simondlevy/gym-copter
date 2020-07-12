#!/usr/bin/env python3
'''
Takeoff-and-hover environment

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class Takeoff(gym.Env, EzPickle):

    FRAMES_PER_SECOND = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()

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

        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics
        state = np.zeros(12)
        self.dynamics.setState(state)

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        d.setMotors(action)
        d.update()

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta, psi, _ = d.getState()

        # Set pose in display
        self.pose = posx, posy, posz, phi, theta, psi

        # Convert state to usable form
        state = np.array([posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta])

        # Reward is a simple penalty for overall takeoff and velocity
        shaping = np.sqrt(posx**2 + posy**2) 
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

        self.prev_shaping = shaping

        done = False

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        from gym_copter.rendering.threed import ThreeDTakeoffRenderer

        # Create renderer if not done yet
        if self.renderer is None:
            self.renderer = ThreeDTakeoffRenderer(self, self.LANDING_RADIUS)

        return self.renderer.render()

    def close(self):

        return

## End of Takeoff class ----------------------------------------------------------------

def constrain_abs(x, lim):

    return -lim if x < -lim else (+lim if x > +lim else x)

def heuristic(env, s, lastError):

    # Extract altitude, vertical velocity from state, negating for NED => ENU
    posz, velz = -s[4:6]

    TARGET = 5

    # PID params
    ALT_P = 1.0
    VEL_P = 1.0
    VEL_D = 0

    dt = 1. / env.FRAMES_PER_SECOND

    # Compute v setpoint and error
    velTarget = (TARGET - posz) * ALT_P
    velError = velTarget - velz

    # Update error integral and error derivative
    deltaError = (velError - lastError) / dt if abs(lastError) > 0 else 0
    lastError = velError

    # Compute control u
    u = VEL_P * velError + VEL_D * deltaError

    u = np.clip(u, -1, +1)

    return u, lastError

def heuristic_takeoff(env, renderer=None, seed=None):

    import time

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    lastError = 0

    while True:

        u, lastError = heuristic(env, state, lastError)
        action = u * np.ones(4)
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

    from gym_copter.rendering.threed import ThreeDTakeoffRenderer
    import threading

    env = Takeoff()

    renderer = ThreeDTakeoffRenderer(env)

    thread = threading.Thread(target=heuristic_takeoff, args=(env, renderer))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start()    
