import logging

import nengo
import numpy as np

# from nengo_fpga.networks import FpgaPesEnsembleNetwork

# Set the nengo logging level to 'info' to display all of the information
# coming back over the ssh connection.
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# ---------------- BOARD SELECT ----------------------- #
# Change this to your desired device name
# board = "de1"
# ---------------- BOARD SELECT ----------------------- #


# Pendulum object. Handles the logic and simulation of the pendulum.
class Pendulum:
    def __init__(
        self,
        mass=1.0,
        length=1.0,
        dt=0.001,
        g=10.0,
        seed=None,
        max_torque=2,
        max_speed=8,
        limit=2.0,
        bounds=None,
    ):
        self.mass = mass
        self.length = length
        self.dt = dt
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.limit = limit
        self.extra_mass = 0
        self.bounds = bounds
        self.reset(seed)

    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-self.limit, self.limit)
        self.dtheta = self.rng.uniform(-1, 1)

    def step(self, u):
        u = np.clip(u, -1, 1) * self.max_torque

        mass = self.mass + self.extra_mass
        self.dtheta += (
            -3 * self.g / (2 * self.length) * np.sin(self.theta + np.pi)
            + 3.0 / (mass * self.length ** 2) * u
        ) * self.dt
        self.theta += self.dtheta * self.dt
        self.dtheta = np.clip(self.dtheta, -self.max_speed, self.max_speed)

        if self.bounds:
            self.theta = np.clip(self.theta, self.bounds[0], self.bounds[1])
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def set_extra_mass(self, mass):
        self.extra_mass = mass

    def generate_html(self, desired):
        len0 = 40 * self.length
        x1 = 50
        y1 = 50
        x2 = x1 + len0 * np.sin(self.theta)
        y2 = y1 - len0 * np.cos(self.theta)
        x3 = x1 + len0 * np.sin(desired)
        y3 = y1 - len0 * np.cos(desired)
        return """
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            <line x1="{x1}" y1="{y1}" x2="{x3}" y2="{y3}" style="stroke:blue"/>
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:black"/>
        </svg>
        """.format(
            x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
        )


class PendulumNetwork(nengo.Network):
    def __init__(self, label=None, **kwargs):
        super(PendulumNetwork, self).__init__(label=label)
        self.env = Pendulum(**kwargs)

        with self:

            def func(t, x):
                self.env.set_extra_mass(x[2])
                self.env.step(x[0])
                func._nengo_html_ = self.env.generate_html(desired=x[1])
                return (self.env.theta, self.env.dtheta)

            self.pendulum = nengo.Node(func, size_in=3, label="Pendulum Obj")

            self.q_target = nengo.Node(None, size_in=1, label="Target Pos")
            nengo.Connection(self.q_target, self.pendulum[1], synapse=None)

            self.u = nengo.Node(None, size_in=1, label="Control Signal")
            nengo.Connection(self.u, self.pendulum[0], synapse=0)
            self.u_extra = nengo.Node(None, size_in=1, label="Adaptive Control Signal")
            nengo.Connection(self.u_extra, self.pendulum[0], synapse=0)

            self.q = nengo.Node(None, size_in=1, label="Pendulum Pos (q)")
            self.dq = nengo.Node(None, size_in=1, label="Pendulum Pos Deriv (dq)")
            nengo.Connection(self.pendulum[0], self.q, synapse=None)
            nengo.Connection(self.pendulum[1], self.dq, synapse=None)

            self.extra_mass = nengo.Node(None, size_in=1, label="Extra Mass")
            nengo.Connection(self.extra_mass, self.pendulum[2], synapse=None)


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

    # PES Ensemble to compute the adaptive control signal to compensate for
    # unknown variables introduced into the environment.
    # adapt_ens = FpgaPesEnsembleNetwork(
    #     board,
    #     n_neurons=1000,
    #     dimensions=1,
    #     learning_rate=1e-5,
    #     function=lambda x: [0],
    #     label="pes ensemble",
    # )

    # Compute the adaptive control signal. The adaptive control signal is
    # computed as a mapping between the current angle of the pendulum, and
    # an additional control signal (u_extra) added to the control signal (u).
    # The error signal used for the adaptive ensemble is simply -u.
    # nengo.Connection(env.q, adapt_ens.input, synapse=None)
    # nengo.Connection(env.u, adapt_ens.error, transform=-1)
    # nengo.Connection(adapt_ens.output, env.u_extra, synapse=None)

    # Extra mass to add to the pendulum. To demonstrate the adaptive
    # controller.
    extra_mass = nengo.Node(None, size_in=1, label="Extra Mass")
    nengo.Connection(extra_mass, env.extra_mass, synapse=None)
