import nengo
import numpy as np


def input_func(t):
    return [np.sin(t * 2*np.pi), np.cos(t * 2*np.pi)]

with nengo.Network() as model:

    # Input stimulus
    input_node = nengo.Node(input_func)

    # "Pre" ensemble of neurons, and connection from the input
    pre = nengo.Ensemble(50, 2)
    nengo.Connection(input_node, pre)

    # "Post" ensemble of neurons, and connection from "Pre"
    post = nengo.Ensemble(50, 2)
    conn = nengo.Connection(pre, post)

    # Create an ensemble for the error signal
    # Error = actual - target = "post" - input
    error = nengo.Ensemble(50, 2)
    nengo.Connection(post, error)
    nengo.Connection(input_node, error, transform=-1)

    # Add the learning rule on the pre-post connection
    conn.learning_rule_type = nengo.PES(learning_rate=1e-4)

    # Connect the error into the learning rule
    nengo.Connection(error, conn.learning_rule)
