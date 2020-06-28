<img src="media/lander2d.gif" width=400 align="left">
<img src="media/lander3d.gif" width=400 align="top">

<p>

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
% python3 setup.py install
% python3 gym_copter/envs/lander2d.py
```
(On Linux you will probably have to run pip3 with <tt>sudo</tt>

You should see the copter land safely.

## Reinforcement learning


To get started, I recommend cloning this
[fork](https://github.com/simondlevy/Deep-Reinforcement-Learning-Hands-On-Second-Edition) 
of the code repository from the excellent
[Deep Reinforcement Learning Hands-On, Second Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks-ebook/dp/B076H9VQH6) book.  Once you've done that (and installed whatever additional
packages you need), you can try out the code from Chapter 19 of the book:

```
% cd Deep-Reinforcement-Learning-Hands-On-Second-Edition/Chapter19
% python3 04_train_ppo.py -e gym_copter:Lander-v2 -n lander
```

This will use a [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) agent on a 2D quadcopter model
that is rewarded for landing successfully between two flags (a reward slightly
over 200).  In my experience this can take a several thousand episodes (10-30
minutes, depending on the speed of your computer).  After that

To play back the best agent at any given step, you can use the
<tt>02\_play.py</tt> script in the Chapter 19 folder:

```
% python3 02_play.py -e gym_copter:Lander-v2 -s 1 -m saves/ppo-lander/best_<REWARD>_<ITERATION>.dat
```

where ```<REWARD>``` is the amount of reward and ```<ITERATION>``` is the number of iterations at which it was saved.
(It is easiest to do this through tab completion.) You should see brief animation of the vehicle landing safely 
(or failing to land safely, if you didn't train to a reward of at least 200).  In the process of creating the movie,
the program will also create an <b>images</b. folder containing a lot of .png (image) files, which you can delete.

In addition to TRPO, the <tt>Chapter19</tt> folder has programs to try other learning agents, including
[A2C](https://arxiv.org/abs/1506.02438), 
[TRPO](https://arxiv.org/abs/1502.05477), 
[ACKTR](https://arxiv.org/abs/1708.05144), 
and [SAC](https://arxiv.org/abs/1801.01290).

## Similar projects

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
