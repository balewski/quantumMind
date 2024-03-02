#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""

Based on https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax/

simple circuit with anstaz, 5 qubits

0: ──RY(M0)──RX(1.00)──RY(1.00)──RX(1.00)─╭●─────────────────────────────────────────────────────
1: ──RY(M1)───────────────────────────────╰X──RX(1.00)──RY(1.00)──RX(1.00)─╭●────────────────────
2: ──RY(M2)────────────────────────────────────────────────────────────────╰X──RX(1.00)──RY(1.00)
3: ──RY(M3)──────────────────────────────────────────────────────────────────────────────────────
4: ──RY(M4)──────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────────────────────────────────────────────╭X─┤ ╭<Z+Z+Z+Z+Z>
──────────────────────────────────────────────────────────────────────────────│──┤ ├<Z+Z+Z+Z+Z>
───RX(1.00)─╭●────────────────────────────────────────────────────────────────│──┤ ├<Z+Z+Z+Z+Z>
────────────╰X──RX(1.00)──RY(1.00)──RX(1.00)─╭●───────────────────────────────│──┤ ├<Z+Z+Z+Z+Z>
─────────────────────────────────────────────╰X──RX(1.00)──RY(1.00)──RX(1.00)─╰●─┤ ╰<Z+Z+Z+Z+Z>


task: optimize params of ansatz to reproduce mapping between 4 x,y pairs

"""

import pennylane as qml
import jax
from jax import numpy as jnp
import optax

n_wires = 5
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.25, 0.1])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    # data embedding
    for i in range(n_wires):
        # data[i] will be of shape (4,); we are
        # taking advantage of operation vectorization here
        qml.RY(data[i], wires=i)

    # trainable ansatz
    for i in range(n_wires):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    # we use a sum of local Z's as an observable since a
    # local Z would only be affected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

######################################################################
# We will define a  cost function that computes the overlap between model output and target data

@jax.jit
def loss_fn(params, data, targets):  # it is chi2/dof
    predictions = my_model(data, params["weights"], params["bias"])
    loss = jnp.sum((targets - predictions) ** 2 / len(data))
    return loss


######################################################################
# Initialize your parameters

weights = jnp.ones([n_wires, 3])
bias = jnp.array(0.)
params = {"weights": weights, "bias": bias}

######################################################################
#  we can see the current loss as well as the parameter gradients:

print('init loss:',loss_fn(params, data, targets))

print('init par grad:',jax.grad(loss_fn)(params, data, targets))

# .... print circuit
print(qml.draw(circuit, decimals=2)(data[0], weights))

######################################################################
# Create the optimizer
# `other available optimizers <https://optax.readthedocs.io/en/latest/api.html>`__

opt = optax.adam(learning_rate=0.3)
opt_state = opt.init(params)

######################################################################
# We first define our ``update_step`` function, which needs to do a couple of things:
#
# -  Compute the loss function (so we can track training) and the gradients (so we can apply an
#    optimization step). We can do this in one execution via the ``jax.value_and_grad`` function.
#
# -  Apply the update step of our optimizer via ``opt.update``
#
# -  Update the parameters via ``optax.apply_updates``
#

def update_step(params, opt_state, data, targets):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


######################################################################
# TRAIN
steps = 200
loss_history = []

for i in range(steps):
    params, opt_state, loss_val = update_step(params, opt_state, data, targets)
    if i % 5 == 0:  print(f"Step: {i} Loss: {loss_val}")
    loss_history.append(loss_val)

print('opt weights & bias:',params,'\n')
# .... print circuit
print(qml.draw(circuit, decimals=2)(data[0], params["weights"]))

######################################################################
# INFER on trained params
#  define a function to make a predictions over multiple data points.

preds = my_model(data, params["weights"], params["bias"])

print('\npred:',preds,type(preds))
print('targets:',targets,type(targets))
res=preds-targets
print('L2:',res**2)
loss = jnp.sum((targets - preds) ** 2 / len(data))
print('loss:',loss,'ndata:',len(data))
