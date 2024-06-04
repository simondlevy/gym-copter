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
% python3 lander.py
```
(On Linux you will may need to run pip3 with <tt>sudo</tt>.)

You should see the copter land safely, using a simple solution (constant
thrust on all motors) to the landing environment provided by gym-copter.  This
can to serve as a basis for comparison with learning algorithms.  

## Similar projects

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
