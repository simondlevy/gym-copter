import nengo
import numpy as np

from pendulum import PendulumNetwork

# Nengo network proper
with nengo.Network(seed=3) as model:

    env = PendulumNetwork(mass=4, max_torque=100, seed=1)

    # The target angle for the pendulum (q)
    q_target = nengo.Node(np.sin, label="Target Pendulum Angle")
    nengo.Connection(q_target, env.q_target, synapse=None)

    # The derivative of the target angle signal (dq)
    dq_target = nengo.Node(None, size_in=1)
    nengo.Connection(q_target, dq_target, synapse=None, transform=1000)
    nengo.Connection(q_target, dq_target, synapse=0, transform=-1000)

    # The difference between the target angle and the actual angle of the
    # pendulum
    q_diff = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(env.q_target, q_diff, synapse=None)
    nengo.Connection(env.q, q_diff, synapse=None, transform=-1)

    # The difference between the target dq and the pendulum's dq
    dq_diff = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(dq_target, dq_diff, synapse=None)
    nengo.Connection(env.dq, dq_diff, synapse=None, transform=-1)

    # Compute the control signal (u) where u = k_p * q + k_d * dq
    k_p = 1.0
    nengo.Connection(q_diff, env.u, transform=k_p, synapse=None)

    k_d = 0.2
    nengo.Connection(dq_diff, env.u, transform=k_d, synapse=None)

    # >>>>>>>>>>>>>>>>>>>  Regular Nengo Code >>>>>>>>>>>>>>>>>>>
    with nengo.Network() as adapt_ens:
        n_neurons = 1000
        dimensions = 1
        learning_rate = 1e-5
        output_func = lambda x: [0]

        # Inputs and outputs from the `adapt_ens` network
        adapt_ens.input = nengo.Ensemble(n_neurons, dimensions)
        adapt_ens.error = nengo.Node(size_in=1)
        adapt_ens.output = nengo.Node(size_in=1)

        # Connection to output node (note, weights initialized to zero using the
        # function argument)
        conn = nengo.Connection(
            adapt_ens.input,
            adapt_ens.output,
            synapse=None,
            function=output_func,
        )

        # Add the learning rule on the output connection
        conn.learning_rule_type = nengo.PES(learning_rate=learning_rate)

        # Connect the error into the learning rule
        nengo.Connection(adapt_ens.error, conn.learning_rule)

    # >>>>>>>>>>>>>>>>>>>  Regular Nengo Code >>>>>>>>>>>>>>>>>>>

    # Compute the adaptive control signal. The adaptive control signal is
    # computed as a mapping between the current angle of the pendulum, and
    # an additional control signal (u_extra) added to the control signal (u).
    # The error signal used for the adaptive ensemble is simply -u.
    nengo.Connection(env.q, adapt_ens.input, synapse=None)
    nengo.Connection(env.u, adapt_ens.error, transform=-1)
    nengo.Connection(adapt_ens.output, env.u_extra, synapse=None)

    # Extra mass to add to the pendulum. To demonstrate the adaptive
    # controller.
    extra_mass = nengo.Node(None, size_in=1, label="Extra Mass")
    nengo.Connection(extra_mass, env.extra_mass, synapse=None)
