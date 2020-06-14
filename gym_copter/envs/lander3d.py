#!/usr/bin/env python3
'''
3D Copter-Lander, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

from gym_copter.envs.rendering.threed import ThreeD, create_axis

class _ThreeDLander(ThreeD):

    def __init__(self, env, radius=2):

        ThreeD.__init__(self, env, lim=5, label='Lander', viewangles=[30,120])

        self.circle = create_axis(self.ax, 'r')
        pts = np.linspace(-np.pi, +np.pi, 1000)
        self.circle_x = radius * np.sin(pts)
        self.circle_y = radius * np.cos(pts)
        self.circle_z = np.zeros(self.circle_x.shape)

    def _animate(self, _):

        ThreeD._animate(self, _)

        self.circle.set_data(self.circle_x, self.circle_y)
        self.circle.set_3d_properties(self.circle_z)

class Lander3D(gym.Env, EzPickle):

    # Parameters to adjust  
    INITIAL_RANDOM_OFFSET = 1.0 # perturbation factor for initial horizontal position
    INITIAL_ALTITUDE      = 5
    LANDING_RADIUS        = 2
    RESTING_DURATION      = 1.0 # for rendering for a short while after successful landing
    FRAMES_PER_SECOND     = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        # Action is three floats [throttle, roll, pitch]
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

        # Support for rendering
        self.pose = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.prev_shaping = None

        self.resting_count = 0

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Initialize custom dynamics with random perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Y] =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        return self.step(np.array([0, 0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed() or self.resting_count:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            t,r,p = (action[0]+1)/2, action[1], action[2]  # map throttle demand from [-1,+1] to [0,1]
            d.setMotors(np.clip([t-r-p, t+r+p, t+r+p, t-r-p], 0, 1))
            d.update(1./self.FRAMES_PER_SECOND)

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = d.getState()[:10]

        # Negate for NED => ENU
        posz  = -posz
        velz  = -velz

        # Set lander pose in display if we haven't landed
        if not (self.dynamics.landed() or self.resting_count):
            self.pose = posx, posy, posz, phi, theta

        # Convert state to usable form
        state = np.array([posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -12 * np.sqrt(np.sum(state[0:6]**2))
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(posy) >= 10:
            done = True
            reward = -100

        elif self.resting_count:

            self.resting_count -= 1

            if self.resting_count == 0:
                done = True

        # It's all over once we're on the ground
        elif self.dynamics.landed():

            # Win bigly we land safely between the flags
            if abs(posy) < self.LANDING_RADIUS: 

                reward += 100

                self.resting_count = int(self.RESTING_DURATION * self.FRAMES_PER_SECOND)

        elif self.dynamics.crashed():

            # Crashed!
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        return True

    def close(self):

        return

    def plotter(self):

        # Pass title to 3D display
        return _ThreeDLander(self)

## End of Lander3D class ----------------------------------------------------------------


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
                  s[8] is the roll angle
                  s[9] is the roll angular speed
     returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    # Angle target
    A = 0.05
    B = 0.06

    # Angle PID
    C = 0.025
    D = 0.05
    E = 0.4

    # Vertical PID
    F = 1.15
    G = 1.33

    posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = s

    phi_targ = posy*A + vely*B              # angle should point towards center
    phi_todo = (phi-phi_targ)*C + phi*D - velphi*E

    theta_targ = posy*A + vely*B            # angle should point towards center
    theta_todo = (theta-theta_targ)*C + theta*D - veltheta*E

    hover_todo = -(posz*F + velz*G)

    return hover_todo, phi_todo, theta_todo

def heuristic_lander(env, plotter=None, seed=None):

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

        if False: #not env.resting_count and (steps % 20 == 0 or done):
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1

        if done: break

        if not plotter is None:
            time.sleep(1./env.FRAMES_PER_SECOND)

    env.close()
    return total_reward


if __name__ == '__main__':

    import threading

    env = Lander3D()

    plotter = env.plotter()

    thread = threading.Thread(target=heuristic_lander, args=(env, plotter, 2))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    plotter.start()    
