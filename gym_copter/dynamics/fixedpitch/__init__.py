'''
Dynamics classes for UAVs with fixed-pitch rotors (e.g., quadcopters)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


class FixedPitchDynamics(MultirotorDynamics):

    def __init__(self, vparams, framesPerSecond):

        MultirotorDynamics.__init__(self, vparams, framesPerSecond)

        self.B = vparams['B']  # thrust coefficient
        self.L = vparams['L']  # arm length

    def _getThrusts(self, u1, u2, u3):

        return self.B * u1, self.L * self.B * u2, self.L * self.B * u3
