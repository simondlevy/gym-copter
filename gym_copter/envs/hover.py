'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np

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
