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

## Reinforcement learning

To use gym-copter in your Reinforcement Learning work, you'll want to install it in the usual way:

```
% python3 setup.py install
```

(On Linux you will probably have to run this command with <tt>sudo</tt>.)

To run the scripts in the [learning](https://github.com/simondlevy/gym-copter/tree/master/learning) folder,
you'll also want to clone and install my
[fork](https://github.com/simondlevy/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
of the code from Deep Reinforcement Learning Hands-On, Second Edition.  Once you've done that, you can
return to the gym-copter repository and do the following:

```
% cd learning
% python3 trpo.py -e Copter-v0 -n altitude
```

This will use a [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) agent on a model
that is rewarded for reaching an altitude of 10 meters.  After a few hundred
thosand iterations or so, the program should should report saving the current
best agent to a file.  As soon as this happens, you can test the agent by
doing:

```
% python3 play.py --render -e Copter-v0 -m saves/trpo-altitude/best_-<REWARD>_<ITER>.dat
```

where ```<REWARD>``` is the amount of reward and ```<ITER>``` is the number of iterations at which it was saved.
(It is easiest to do this through tab completion.) You should see brief animation of the vehicle rising to
10 meters altitude.

In addition to TRPO, the <tt>learning</tt> folder has program to try other learning agents, including
[A2C](https://arxiv.org/abs/1506.02438), 
[ACKTR](https://arxiv.org/abs/1708.05144), 
[PPO](https://arxiv.org/abs/1707.06347), 
and [SAC](https://arxiv.org/abs/1801.01290).

## Similar projects

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
