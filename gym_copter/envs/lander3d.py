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

    # World size in meters
    SIZE = 10

    # Scaling factor for angular velocities
    ANGLE_VEL_SCALE = 20

    # Perturbation factor for initial horizontal position
    INITIAL_RANDOM_OFFSET = 0.0 

    # Update rate
    FPS = 50

    # Radius for landing
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

        # Compute time constant from update rate
        self.dt = 1./self.FPS

        # Observation space is dynamics state space without yaw and its derivative.
        # Useful interval is -1 .. +1, but spikes can be higher.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        # Action space is throttle, roll, pitch demands
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)

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

        return self.step(np.array([0, 0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed():
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            # map throttle demand from [-1,+1] to [0,1]
            t, r, p = (action[0]+1)/2, action[1], action[2] 
            d.setMotors(np.clip([t-r-p, t+r+p, t+r-p, t-r+p], 0, 1))
            d.update(self.dt)

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = d.getState()[:10]

        # Set lander pose in display if we haven't landed
        if not self.dynamics.landed():
            self.pose = posx, posy, posz

        # Convert state to usable form
        posx /= self.SIZE
        velx *= self.SIZE * self.dt
        posy /= self.SIZE
        vely *= self.SIZE * self.dt
        posz /= -self.SIZE
        velz *= -self.SIZE * self.dt
        velphi *= self.ANGLE_VEL_SCALE * self.dt
        veltheta *= self.ANGLE_VEL_SCALE * self.dt
        state = np.array([posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -100 * np.sqrt(np.sum(state[0:6]**2))

        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0
                                                                  
        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(posx) >= 1.0 or abs(posy) >= 1.0:
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

    # Angle target
    A = 0.5
    B = 3

    # Angle PID
    C = 0.025
    D = 0.05

    # Vertical target
    E = 0.8 

    # Vertical PID
    F = 7.5
    G = 10

    posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = s

    phi_targ = posx*A + velx*B         
    phi_todo = (phi-phi_targ)*C + velphi*D

    theta_targ = posy*A + vely*B         
    theta_todo = (theta-theta_targ)*C + veltheta*D

    hover_targ = E*np.sqrt(posx**2+posy**2)
    hover_todo = (hover_targ - posz)*F - velz*G

    return hover_todo, phi_todo, theta_todo

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
