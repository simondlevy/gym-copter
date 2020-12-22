<img src="media/lander3d.gif" height=250 align="left">
<img src="media/lander2d.gif" height=180 align="top">

<br><br>

# gym-copter
OpenAI Gym environment for reinforcement learning with multicopters.  

## Features:

* Pure Python / Cross-platform

* Uses realistic multirotor dynamics
([Bouabdallah et al. 2004](https://infoscience.epfl.ch/record/97532/files/325.pdf)) that can be
subclassed for a particular vehicle configuration (quad, hex, octo, etc.)

* Supports rendering via a Heads-Up Display (HUD) similar to Mission Planner / QGroundControl.

## Quickstart

```
% pip3 install gym
% python3 setup.py install
% python3 gym_copter/envs/lander2d.py
```
(On Linux you will probably have to run pip3 with <tt>sudo</tt>

You should see the copter land safely.

## Evolving a neural controller

The [NEAT](https://github.com/simondlevy/gym-copter/tree/master/neat)
sub-folder of this repository shows how you can use the NEAT algorithm to
evolve a neural controller for your copter.

## Supported environments

* **Lander-v0** 2D [LunarLander](https://gym.openai.com/envs/LunarLander-v2/)-style challenge

* **Lander3D-v0** 3D lander with reward proportional on proximity of touchdown to center

* **Lander3DHardcore-v0** 3D lander with maximum reward for landing inside circle

## Similar projects

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
