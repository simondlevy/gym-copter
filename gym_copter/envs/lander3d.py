#!/usr/bin/env python3
"""
Copter-Lander, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class CopterLander3D(gym.Env, EzPickle):

    # Perturbation factor for initial horizontal position
    INITIAL_RANDOM_OFFSET = 1.5

    FPS = 50

    SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

    LANDING_RADIUS = 2

    # Rendering properties ---------------------------------------------------------

    # For rendering for a short while after successful landing
    RESTING_DURATION = 50

    VIEWPORT_W = 600
    VIEWPORT_H = 400

    # -------------------------------------------------------------------------------------

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        # Ground level
        self.ground_z = 0

        # Support for rendering
        self.angle = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.prev_shaping = None

        self.resting_count = 0

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Set its landing altitude
        self.dynamics.setGroundLevel(self.ground_z)

        # Initialize custom dynamics with random perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_Y] =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Z] = -self.VIEWPORT_H/self.SCALE
        self.dynamics.setState(state)

        return self.step(np.array([0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed() or self.resting_count:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            t,r = (action[0]+1)/2, action[1]  # map throttle demand from [-1,+1] to [0,1]
            d.setMotors(np.clip([t-r, t+r, t+r, t-r], 0, 1))
            d.update(1./self.FPS)

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = d.getState()[:10]

        # Negate for NED => ENU
        posz  = -posz
        velz  = -velz

        # Set lander pose in display if we haven't landed
        if not (self.dynamics.landed() or self.resting_count):
            self.pose = posy, posz, -phi

        # Convert state to usable form
        state = np.array([
            posy / (self.VIEWPORT_W/self.SCALE/2),
            (posz - (self.ground_z)) / (self.VIEWPORT_H/self.SCALE/2),
            vely*(self.VIEWPORT_W/self.SCALE/2)/self.FPS,
            velz*(self.VIEWPORT_H/self.SCALE/2)/self.FPS,
            phi,
            20.0*velphi/self.FPS
            ])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -100 * np.sqrt(np.sum(state[0:4]**2))
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(state[0]) >= 1.0:
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

                self.resting_count = self.RESTING_DURATION

        elif self.dynamics.crashed():

            # Crashed!
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        return True

    def close(self):

        return

# End of CopterLander3D class ----------------------------------------------------------------


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    # Angle target
    A = 0.5
    B = 3

    # Angle PID
    C = 0.025
    D = 0.05

    # Vertical target
    E = 0.8 

    # Vertical PID
    F = 7.5#10
    G = 10

    angle_targ = s[0]*A + s[2]*B         # angle should point towards center
    angle_todo = (s[4]-angle_targ)*C + s[5]*D

    hover_targ = E*np.abs(s[0])           # target y should be proportional to horizontal offset
    hover_todo = (hover_targ - s[1])*F - s[3]*G

    return hover_todo, angle_todo

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    np.random.seed(seed)
    total_reward = 0
    steps = 0
    state = env.reset()
    while True:
        action = heuristic(env,state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            still_open = env.render()
            if not still_open: break

        if not env.resting_count and (steps % 20 == 0 or done):
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done: break

    env.close()
    return total_reward


if __name__ == '__main__':

    demo_heuristic_lander(CopterLander3D(), seed=None, render=True)
