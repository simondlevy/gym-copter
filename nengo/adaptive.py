'''
Nengo adaptive controller with PES learning

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo


class PlantNetwork(nengo.Network):

    def __init__(self, plant, label=None, **kwargs):

        nengo.Network.__init__(self, label=label)

        self.env = plant(**kwargs)

        with self:

            def func(t, x):
                self.env.set_extra_force(x[2])
                self.env.step(x[0])
                func._nengo_html_ = self.env.generate_html(desired=x[1])
                return (self.env.theta, self.env.dtheta)

            self.plant = nengo.Node(func, size_in=3, label="Object")

            self.q_target = nengo.Node(None, size_in=1, label="Target")
            nengo.Connection(self.q_target, self.plant[1], synapse=None)

            self.u = nengo.Node(None, size_in=1, label="Control Signal")
            nengo.Connection(self.u, self.plant[0], synapse=0)
            self.u_extra = nengo.Node(None, size_in=1,
                                      label="Adaptive Control Signal")
            nengo.Connection(self.u_extra, self.plant[0], synapse=0)

            self.q = nengo.Node(None, size_in=1, label="Pos (q)")
            self.dq = nengo.Node(None, size_in=1, label="Pos Deriv (dq)")
            nengo.Connection(self.plant[0], self.q, synapse=None)
            nengo.Connection(self.plant[1], self.dq, synapse=None)

            self.extra_force = nengo.Node(None, size_in=1, label="Extra Force")
            nengo.Connection(self.extra_force, self.plant[2], synapse=None)
