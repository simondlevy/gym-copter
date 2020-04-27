'''
Third-Person (3D) view using matplotlib for vehicle and target

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.tpv import TPV, _Vehicle

class TpvTarget(TPV):

    def __init__(self, env):

        TPV.__init__(self, env)

        self.target = _Vehicle(self.ax)
        
    def _animate(self, _):

       TPV._animate(self, _)

       # Update the target animation
       self.target.update(*self.env.state[12:])
