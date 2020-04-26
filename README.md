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
% python3 examples/takeoff.py --render
```

You should see a Heads-Up Display (HUD) of the vehicle rising to 10 meters altitude.  If you omit the
```--render``` you'll get a plot of the altitude, iassociated reward, vertical velocity, and motor actions.

## Reinforcement learning

To use gym-copter in your Reinforcement Learning work, you'll want to install it in the usual way:

```
% python3 setup.py install
```

(On Linux you will probably have to run this command with <tt>sudo</tt>.)

To get started, I recommend cloning this
[repository](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
of the code from the excellent
[Deep Reinforcement Learning Hands-On, Second Edition](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks-ebook/dp/B076H9VQH6) book.  Once you've done that (and installed whatever additional
packages you need), you can try out the code from Chapter 19 of the book:

```
% cd Deep-Reinforcement-Learning-Hands-On-Second-Edition/Chapter19
% python3 03_train_trpo.py -e gym_copter:Copter-v0 -n takeoff
```

This will use a [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) agent on a model
that is rewarded for reaching an altitude of 10 meters.  After a few hundred
thousand iterations or so, the program should should report saving the current
best agent to a file.  

To play back this best agent (and subsequent ones), you can use the <tt>02\_play.py</tt> script in
the Chapter 19 folder:

```
% python3 02_play.py -e gym_copter:Copter-v0 -m saves/trpo-altitude/best_-<REWARD>_<ITER>.dat -r copter-v0
```

where ```<REWARD>``` is the amount of reward and ```<ITER>``` is the number of iterations at which it was saved.
(It is easiest to do this through tab completion.) You should see brief animation of the vehicle rising to
10 meters altitude.  A new folder <tt>copter-v0</tt> will contain an mp4 copy of the animation.

You can also try to learn a second task, covering the maximum distance, by doing:

```
% python3 trpo.py -e Copter-v1 -n distance
```
In addition to TRPO, the <tt>Chapter19</tt> folder has programs to try other learning agents, including
[A2C](https://arxiv.org/abs/1506.02438), 
[ACKTR](https://arxiv.org/abs/1708.05144), 
[PPO](https://arxiv.org/abs/1707.06347), 
and [SAC](https://arxiv.org/abs/1801.01290).

## Third-person (3D) view

Gym-copter also supports a third-person (3D) view for rendering.  To try it out:

```
% python3 examples/leap.py
```

This simulation runs the motors at full speed until the vehicle reaches an altitude of 10 meters, after which the front motors
are reduced to half speed, causing the vehicle to fly forward into a dive.

## Similar projects

[gym\_rotor](https://github.com/inkyusa/gym_rotor)

[GymFC](https://github.com/wil3/gymfc)

[How to Train Your Quadcopter](https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61)
