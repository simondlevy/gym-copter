# Learning a neural controller

The code in this directory allows you to try out various Deep Reinforcement Learning (DRL) algorithms
with gym-copter.

To try this out for yourself, you should do the following:

1. Clone and install [AC-Gym](https://github.com/simondlevy/AC-Gym).  

2. Run ```python3 [DIR]/td3-learn.py --env gym_copter:Lander-v0 --target 200```

where ```[DIR``` is the directory in which you put AC-Gym; for example:

```python3 /home/levy/AC-Gym/td3-learn.py --env gym_copter:Lander-v0 --target 200```

This will run the [TD3](https://arxiv.org/pdf/1802.09477.pdf) algorithm on the 2D landing task.

Once learning finishes, you can test out your evolved network by doing:

```
% python3 [DIR]/ac-test.py models/gym_copter:Lander-v0/<fitness>.dat
```

where ```<fitness>``` is the fitness of your evolved network.
