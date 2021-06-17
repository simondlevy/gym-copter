'''
Nengo adaptive controller with PES learning

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import numpy as np


class PlantNetwork(nengo.Network):

    def __init__(self, env, label, **kwargs):

        nengo.Network.__init__(self, label=label)

        self.env = env(**kwargs)

        with self:

            def func(t, x):
                self.env.set_extra_force(x[2])
                y, dy = self.env.step(x[0])
                func._nengo_html_ = self.env.generate_html(desired=x[1])
                return y, dy

            self.plant = nengo.Node(func, size_in=3, label=(label + ' Object'))

            self.q_target = nengo.Node(None, size_in=1, label='Target')
            nengo.Connection(self.q_target, self.plant[1], synapse=None)

            self.u = nengo.Node(None, size_in=1, label='Control Signal')
            nengo.Connection(self.u, self.plant[0], synapse=0)
            self.u_extra = nengo.Node(None, size_in=1,
                                      label='Adaptive Control Signal')
            nengo.Connection(self.u_extra, self.plant[0], synapse=0)

            self.q = nengo.Node(None, size_in=1, label='Pos (q)')
            self.dq = nengo.Node(None, size_in=1, label='Pos Deriv (dq)')
            nengo.Connection(self.plant[0], self.q, synapse=None)
            nengo.Connection(self.plant[1], self.dq, synapse=None)

            self.extra_force = nengo.Node(None, size_in=1, label='Extra Force')
            nengo.Connection(self.extra_force, self.plant[2], synapse=None)


def run(env, name, q_name, force_name):

    net = PlantNetwork(env, name, seed=1)

    # The target (q)
    q_target = nengo.Node(np.sin, label=('Target ' + name + q_name))
    nengo.Connection(q_target, net.q_target, synapse=None)

    # The derivative of the target angle signal (dq)
    dq_target = nengo.Node(None, size_in=1, label='dq')
    nengo.Connection(q_target, dq_target, synapse=None, transform=1000)
    nengo.Connection(q_target, dq_target, synapse=0, transform=-1000)

    # The difference between the target and the actual
    q_diff = nengo.Ensemble(n_neurons=100, dimensions=1, label='q_diff')
    nengo.Connection(net.q_target, q_diff, synapse=None)
    nengo.Connection(net.q, q_diff, synapse=None, transform=-1)

    # The difference between the target dq and the actual dq
    dq_diff = nengo.Ensemble(n_neurons=100, dimensions=1, label='dq_diff')
    nengo.Connection(dq_target, dq_diff, synapse=None)
    nengo.Connection(net.dq, dq_diff, synapse=None, transform=-1)

    # Compute the control signal (u) where u = k_p * q + k_d * dq
    k_p = 1.0
    nengo.Connection(q_diff, net.u, transform=k_p, synapse=None)

    k_d = 0.2
    nengo.Connection(dq_diff, net.u, transform=k_d, synapse=None)

    def zero(x):
        return [0]

    with nengo.Network() as adapt_ens:

        n_neurons = 1000
        dimensions = 1
        learning_rate = 1e-5
        output_func = zero

        # Inputs and outputs from the `adapt_ens` network
        adapt_ens.input = nengo.Ensemble(n_neurons, dimensions)
        adapt_ens.error = nengo.Node(size_in=1)
        adapt_ens.output = nengo.Node(size_in=1)
        adapt_ens.label = 'pes ensemble'

        # Connection to output node (note, weights initialized to zero using
        # the function argument)
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

    # Compute the adaptive control signal. The adaptive control signal is
    # computed as a mapping between the current value of the system and
    # an additional control signal (u_extra) added to the control signal (u).
    # The error signal used for the adaptive ensemble is simply -u.
    nengo.Connection(net.q, adapt_ens.input, synapse=None)
    nengo.Connection(net.u, adapt_ens.error, transform=-1)
    nengo.Connection(adapt_ens.output, net.u_extra, synapse=None)

    # Extra force to add to the system to demonstrate the adaptive controller
    extra_force = nengo.Node(None, size_in=1, label=force_name)
    nengo.Connection(extra_force, net.extra_force, synapse=None)

    return net
