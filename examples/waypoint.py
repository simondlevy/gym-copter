#!/usr/bin/env python3
'''
Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
import gym_copter

from pidctrl import AltitudePidController

# Create and initialize copter environment
env = gym.make('CopterWaypoint-v0')

exit(0)

env.reset()

# Create PID controller
pid  = AltitudePidController(ALTITUDE_TARGET, ALT_P, VEL_P, VEL_I, VEL_D)

# Initialize arrays for plotting
tvals = []
uvals = []
zvals = []
vvals = []
rvals = []

# Motors are initially off
u = -1

# Loop for specified duration
while True:

    # Get current time from environment
    t = env.time()

    # Stop if time excedes duration
    if t > DURATION: break

    # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
    s, r, _, _ = env.step([u])

    # Display the environment
    if args.render: 
        env.render()

    # Extract altitude, vertical velocity from state
    z, v = s

    # Get correction from PID controller
    u = pid.u(z, v, env.dt)

    # Convert u from [0,1] to [-1,+1]
    u = 2 * u - 1

    # Constrain correction to [-1,+1]
    u = max(-1, min(+1, u))

    # Track values
    tvals.append(t)
    uvals.append(u)
    zvals.append(z)
    vvals.append(v)
    rvals.append(r)

# Plot results
if not args.render:
    _subplot(tvals, rvals, 1, 'Reward')
    _subplot(tvals, zvals, 2, 'Altitude (m)')
    _subplot(tvals, vvals, 3, 'Velocity (m/s)')
    _subplot(tvals, uvals, 4, 'Action')
    plt.ylim([-1.1,+1.1])
    plt.show()

# Cleanup
del env
