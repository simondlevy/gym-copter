'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
from numpy import radians
from time import sleep
import argparse
from argparse import ArgumentDefaultsHelpFormatter

import gym
from gym import spaces
from gym.utils import EzPickle, seeding

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics


class _Lander(gym.Env, EzPickle):

    # Physics
    INITIAL_RANDOM_FORCE = 30
    INITIAL_ALTITUDE = 10
    TARGET_RADIUS = 2
    BOUNDS = 10
    FRAMES_PER_SECOND = 50

    # Reward shaping
    OUT_OF_BOUNDS_PENALTY = 100
    INSIDE_RADIUS_BONUS = 100
    MAX_ANGLE = 45
    YAW_PENALTY_FACTOR = 50
    MOTOR_PENALTY_FACTOR = 0.03
    XYZ_PENALTY_FACTOR = 25

    # PIDs for heuristic demo
    PID_X = 0.1
    PID_DX = 0.1
    PID_PHI = 0.05
    PID_DPHI = 0.4
    PID_Z = 1.15
    PID_DZ = 1.33

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FRAMES_PER_SECOND
    }

    def __init__(self, observation_size, action_size):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.pose = None
        self.prev_reward = None
        self.action_size = action_size

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf,
                                            +np.inf,
                                            shape=(observation_size,),
                                            dtype=np.float32)

        # Action is two floats [throttle, roll]
        self.action_space = spaces.Box(-1,
                                       +1,
                                       (action_size,),
                                       dtype=np.float32)

        # Pre-convert max-angle degrees to radians
        self.max_angle = np.radians(self.MAX_ANGLE)

    def seed(self, seed=None):

        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        motors = np.zeros(4)

        # Stop motors after safe landing
        if status == d.STATUS_LANDED:
            d.setMotors(motors)
            self.spinning = False

        # In air, set motors from action
        else:
            motors = np.clip(action, 0, 1)    # stay in interval [0,1]
            d.setMotors(self._get_motors(motors))
            self.spinning = sum(motors) > 0
            d.update()

        # Get new state from dynamics
        state = np.array(d.getState())

        # Extract components from state
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi = state

        # Set pose for display
        self.pose = x, y, z, phi, theta, psi

        # Get penalty based on state and motors
        shaping = -(self.XYZ_PENALTY_FACTOR*np.sqrt(np.sum(state[0:6]**2)) +
                    self.YAW_PENALTY_FACTOR*np.sqrt(np.sum(state[10:12]**2)) +
                    self.MOTOR_PENALTY_FACTOR*np.sum(motors))

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(x) >= self.BOUNDS or abs(y) >= self.BOUNDS:
            done = True
            reward -= self.OUT_OF_BOUNDS_PENALTY

        # Lose bigly for excess roll or pitch
        elif abs(phi) >= self.max_angle or abs(theta) >= self.max_angle:
            done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        else:

            # It's all over once we're on the ground
            if status == d.STATUS_LANDED:

                done = True
                self.spinning = False

                # Win bigly we land safely between the flags
                if np.sqrt(x**2+y**2) < self.TARGET_RADIUS:

                    reward += self.INSIDE_RADIUS_BONUS

            elif status == d.STATUS_CRASHED:

                # Crashed!
                done = True
                self.spinning = False

        # Extract 2D or 3D components of state and rerturn them with the rest
        return (np.array(self._get_state(state), dtype=np.float32),
                reward,
                done,
                {})

    def demo_heuristic(self, seed=None, nopid=False, csvfilename=None):
        '''
        csvfile arg will only be added by 3D scripts.
        '''

        self.seed(seed)
        np.random.seed(seed)

        total_reward = 0
        steps = 0
        state = self.reset()

        dt = 1. / self.FRAMES_PER_SECOND

        actsize = self.action_space.shape[0]

        csvfile = None
        if csvfilename is not None:
            csvfile = open(csvfilename, 'w')
            csvfile.write('t,' + ','.join([('m%d' % k)
                                          for k in range(1, actsize+1)]))
            csvfile.write(',X,dX,y,dY,Z,dZ,phi,dPhi,theta,dTheta\n')

        while True:

            action = self.heuristic(state, nopid)
            state, reward, done, _ = self.step(action)
            total_reward += reward

            if csvfile is not None:

                csvfile.write('%f,' % (dt * steps))

                csvfile.write('%f,%f,%f,%f,' % tuple(action))

                csvfile.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' %
                              tuple(state))

            self.render('rgb_array')

            sleep(1./self.FRAMES_PER_SECOND)

            if (steps % 20 == 0) or done:
                print('steps =  %03d    total_reward = %+0.2f' %
                      (steps, total_reward))

            steps += 1

            if done:
                break

        sleep(1)
        self.close()
        if csvfile is not None:
            csvfile.close()
        return total_reward

    def _reset(self, pose=(0, 0, INITIAL_ALTITUDE, 0, 0), perturb=True):

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.prev_shaping = None

        # Create dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Set up initial conditions
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] = pose[0]
        state[d.STATE_Y] = pose[1]
        state[d.STATE_Z] = -pose[2]  # NED
        state[d.STATE_PHI] = radians(pose[3])
        state[d.STATE_THETA] = radians(pose[4])
        self.dynamics.setState(state)

        # Perturb with a random force
        if perturb:
            self.dynamics.perturb(np.array([self._randforce(),  # X
                                            self._randforce(),  # Y
                                            self._randforce(),  # Z
                                            0,                  # phi
                                            0,                  # theta
                                            0]))                # psi

        # Return initial state
        return self.step(np.zeros(self.action_size))[0]

    def _angle_pid(self, x, dx, phi, dphi):

        phi_targ = x*self.PID_X + dx*self.PID_DX

        return ((phi-phi_targ)*self.PID_TARG
                + phi*self.PID_PHI - dphi*self.PID_DPHI)

    def _hover_pid(self, z, dz):

        return z*self.PID_Z + dz*self.PID_DZ

    def _randforce(self):

        return np.random.uniform(-self.INITIAL_RANDOM_FORCE,
                                 + self.INITIAL_RANDOM_FORCE)


def _make_parser():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')

    parser.add_argument('--nopid', action='store_true',
                        help='Turn off lateral PID control')

    return parser
