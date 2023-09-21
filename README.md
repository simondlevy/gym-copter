<img src="media/lander3d.gif" height=250 align="left">

<br><br>

# gym-copter
Gymnasium environment for reinforcement learning with multicopters, as described 
[here](https://simondlevy.academic.wlu.edu/files/publications/LM2020_011_final_v2.pdf).

## Features:

* Pure Python / Cross-platform

* Uses realistic multirotor dynamics
([Bouabdallah et al. 2004](https://infoscience.epfl.ch/record/97532/files/325.pdf)) 

* Supports 3D rendering

## Dependencies 

* numpy

* matplotlib

* [gymnasium](https://pypi.org/project/gymnasium/)

## Quickstart

```
% pip3 install -e .
% python3 heuristic/lander.py
```
(On Linux you will probably need to run pip3 with <tt>sudo</tt>.)

You should see the copter land safely.

## Evolving a neural controller

The [NEAT](https://github.com/simondlevy/gym-copter/tree/master/neat)
sub-folder of this repository shows how you can use the NEAT algorithm to
evolve a neural controller for your copter.

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
