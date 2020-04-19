<img src="hud.gif" width=500>

# gym-copter
Gym environment for reinforcement learning with multicopters.  

## Features:

* Pure Python / Cross-platform

* Uses realistic multirotor dynamics
([Bouabdallah et al. 2004](https://infoscience.epfl.ch/record/97532/files/325.pdf)) that can be
subclassed for a particular vehicle configuration (quad, hex, octo, etc.)

* Supports rendering via a Heads-Up Display (HUD) similar to Mission Planner / QGroundControl.

## Quickstart

```
% pip3 install gym
% python3 examples/takeoff.py
```

This should pop up the HUD and show the copter rising to an altitude of 10 meters.

## Going further

To use gym-copter in your Reinforcement Learning work, you'll want to install it in the usual way:

```
% python3 setup.py install
```

(On Linux you will probably have to run this command with <tt>sudo</tt>.)

## Similar projects

[gym_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
