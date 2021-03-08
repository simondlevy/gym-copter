'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
from numpy import radians

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics
from gym_copter.pidcontrollers import AltitudeHoldPidController
from gym_copter.envs.task import _Task


class _Hover(_Task):

    BOUNDS = 10
    MAX_STEPS = 1000

    def __init__(self, observation_size, action_size):

        _Task.__init__(self, observation_size, action_size)

        # Set up altitude-hold PID controller for heuristic demo
        self.altpid = AltitudeHoldPidController()

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

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

        # Simple reward: 1 point per successful step
        # (no crash / out-of-bounds)
        reward = 1

        # Assume we're not done yet
        self.done = False

        # Lose bigly if we go outside window
        if abs(x) >= self.BOUNDS or abs(y) >= self.BOUNDS:
            self.done = True
            reward -= self.OUT_OF_BOUNDS_PENALTY

        # Lose bigly for excess roll or pitch
        elif abs(phi) >= self.max_angle or abs(theta) >= self.max_angle:
            self.done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        # It's all over if we crash
        elif status == d.STATUS_CRASHED:

            # Crashed!
            self.done = True
            self.spinning = False

        # Don't run forever!
        elif self.steps == self.MAX_STEPS:

            self.done = True

        self.steps += 1

        # Extract 2D or 3D components of state and rerturn them with the rest
        return (np.array(self._get_state(state), dtype=np.float32),
                reward,
                self.done,
                {})

    def _reset(self, pose=(0, 0, _Task.INITIAL_ALTITUDE, 0, 0), perturb=True):

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.done = False

        # Support for reward shaping
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

        # No steps or reward yet
        self.steps = 0

        # Return initial state
        return self.step(np.zeros(self.action_size))[0]

    def _randforce(self):

        return np.random.uniform(-self.INITIAL_RANDOM_FORCE,
                                 + self.INITIAL_RANDOM_FORCE)
