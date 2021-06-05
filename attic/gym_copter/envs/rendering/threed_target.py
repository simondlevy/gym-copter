'''
Third-Person (3D) view using matplotlib for vehicle and target

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.threed import ThreeD, _Vehicle

class ThreeDTarget(ThreeD):

    def __init__(self, env):

        ThreeD.__init__(self, env)

        self.target = _Vehicle(self.ax, 'r')
        
    def _animate(self, _):

       ThreeD._animate(self, _)

       # Update the target animation
       self.target.update(*self.env.state[12:])
