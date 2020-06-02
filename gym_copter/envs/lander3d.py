#!/usr/bin/env python3
"""
Copter-Lander, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""

import numpy as np
import time

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class CopterLander3D(gym.Env, EzPickle):

    # Perturbation factor for initial horizontal position
    INITIAL_RANDOM_OFFSET = 0.0 

    FPS = 50

    RADIUS = 5

    # For rendering for a short while after successful landing
    RESTING_DURATION = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()

        self.dt = 1./self.FPS

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        # Starting position
        self.startpos = 0, 0, 13.333

        # Support for rendering
        self.pose = None
        self.tpv = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.prev_shaping = None

        self.resting_count = 0

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Initial random perturbation of horizontal position
        xoff, yoff = self.INITIAL_RANDOM_OFFSET * np.random.randn(2)

        # Initialize custom dynamics with perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] =  self.startpos[0] + xoff # 3D copter Y comes from 3D copter X
        state[d.STATE_Y] =  self.startpos[1] + yoff # 3D copter Y comes from 3D copter X
        state[d.STATE_Z] = -self.startpos[2]        # 3D copter Z comes from 3D copter Y, negated for NED
        self.dynamics.setState(state)

        # By showing props periodically, we can emulate prop rotation
        self.props_visible = 0

        return self.step(np.array([0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed():
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            throttle = (action[0] + 1) / 2  # map throttle demand from [-1,+1] to [0,1]
            roll = action[1]
            d.setMotors(np.clip([throttle-roll, throttle+roll, throttle+roll, throttle-roll], 0, 1))
            d.update(self.dt)

        # Get new state from dynamics
        x = d.getState()

        # Parse out state into elements
        posx  =  x[d.STATE_X]
        posy  =  x[d.STATE_Y]
        posz  = -x[d.STATE_Z] 
        velx  =  x[d.STATE_X_DOT]
        vely  =  x[d.STATE_Y_DOT]
        velz  = -x[d.STATE_Z_DOT]
        phi   =  x[d.STATE_PHI]
        velphi = x[d.STATE_PHI_DOT]

        # Set lander pose in display if we haven't landed
        if not self.dynamics.landed():
            self.pose = posx, posy, -posz

        # Convert state to usable form
        state = (
            posx / 10, 
            posz / 6.67, 
            velx * 10 * self.dt,
            velz * 6.67 * self.dt,
            phi,
            20.0 * velphi * self.dt
            )

        # Shape the reward
        reward = 0
        shaping = 0
        shaping -= 100*np.sqrt(state[0]**2 + state[1]**2)  # Lose points for altitude and vertical drop rate'
        shaping -= 100*np.sqrt(state[2]**2 + state[3]**2)  # Lose points for distance from X center and horizontal velocity
                                                                  
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
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

            # Win bigly we land close to the center of the circle
            if np.sqrt(posx**2 + posy**2) < self.RADIUS:

                reward += 100

                self.resting_count = self.RESTING_DURATION

        elif self.dynamics.crashed():

            # Crashed!
            done = True

        time.sleep(self.dt)

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        return True

    def close(self):

        return

    def tpvplotter(self):

        from gym_copter.envs.rendering.tpv import TPV

        # Pass title to 3D display
        return TPV(self, 'Lander')

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

def heuristic_lander(env, plotter, seed=None, render=False):

    # Seed random number generators
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
    plotter.close()
    return total_reward

if __name__ == '__main__':

    import threading

    env = CopterLander3D()

    # Run simulation on its own thread
    plotter = env.tpvplotter()
    thread = threading.Thread(target=heuristic_lander, args=(env,plotter))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    plotter.start()
