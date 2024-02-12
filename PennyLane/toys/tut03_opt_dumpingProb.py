#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

r"""

Based on https://pennylane.ai/qml/demos/tutorial_noisy_circuits/

Noisy circuits
==============

.. meta::
    :property="og:description": Learn how to simulate noisy quantum circuits
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets//N-Nisq.png

.. related::

    tutorial_noisy_circuit_optimization Optimizing noisy circuits with Cirq
    pytorch_noise PyTorch and noisy devices

*Author: Juan Miguel Arrazola — Posted: 22 February 2021. Last updated: 08 April 2021.*

In this demonstration, you'll learn how to simulate noisy circuits using built-in functionality in
PennyLane. We'll cover the basics of noisy channels and density matrices, then use example code to
simulate noisy circuits. PennyLane, the library for differentiable quantum computations, has
unique features that enable us to compute gradients of noisy channels. We'll also explore how
to employ channel gradients to optimize noise parameters in a circuit.

We're putting the N in NISQ.

.. figure:: ../_static/demonstration_assets/noisy_circuits/N-Nisq.png
    :align: center
    :width: 20%

    ..
"""

##############################################################################
#
# Noisy operations
# ----------------
# Noise is any unwanted transformation that corrupts the intended
# output of a quantum computation. It can be separated into two categories.
#
# * **Coherent noise** is described by unitary operations that maintain the purity of the
#   output quantum state. A common source are systematic errors originating from
#   imperfectly-calibrated devices that do not exactly apply the desired gates, e.g., applying
#   a rotation by an angle :math:`\phi+\epsilon` instead of :math:`\phi`.
#
# * **Incoherent noise** is more problematic: it originates from a quantum computer
#   becoming entangled with the environment, resulting in mixed states — probability
#   distributions over different pure states. Incoherent noise thus leads to outputs that are
#   always random, regardless of what basis we measure in.
#
# Mixed states are described by `density matrices
# <https://en.wikipedia.org/wiki/Density_matrices>`__.
# They provide a more general method of describing quantum states that elegantly
# encodes a distribution over pure states in a single mathematical object.
# Mixed states are the most general description of a quantum state, of which pure
# states are a special case.
#
# The purpose of PennyLane's ``default.mixed`` device is to provide native
# support for mixed states and for simulating noisy computations. Let's use ``default.mixed`` to
# simulate a simple circuit for preparing the
# Bell state :math:`|\psi\rangle=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)`. We ask the QNode to
# return the expectation value of :math:`Z_0\otimes Z_1`:
#
import pennylane as qml
from jax import numpy as np
import jax
import jaxopt

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

dev = qml.device('default.mixed', wires=2)


######################################################################
# As before, the output deviates from the desired value as the amount of
# noise increases.
# Modelling the noise that occurs in real experiments requires careful consideration.
# PennyLane
# offers the flexibility to experiment with different combinations of noisy channels to either mimic
# the performance of quantum algorithms when deployed on real devices, or to explore the effect
# of more general quantum transformations.
#
# Channel gradients
# -----------------
#
# The ability to compute gradients of any operation is an essential ingredient of 
# :doc:`quantum differentiable programming </glossary/quantum_differentiable_programming>`.
# In PennyLane, it is possible to
# compute gradients of noisy channels and optimize them inside variational circuits.
# PennyLane supports analytical
# gradients for channels whose Kraus operators are proportional to unitary
# matrices [#johannes]_. In other cases, gradients are evaluated using finite differences.
#
# To illustrate this property, we'll consider an elementary example. We aim to learn the noise
# parameters of a circuit in order to reproduce an observed expectation value. So suppose that we
# run the circuit to prepare a Bell state
# on a hardware device and observe that the expectation value of :math:`Z_0\otimes Z_1` is
# not equal to 1 (as would occur with an ideal device), but instead has the value 0.7781. In the
# experiment, it is known that the
# major source of noise is amplitude damping, for example as a result of photon loss.
# Amplitude damping projects a state to :math:`|0\rangle` with probability :math:`p` and
# otherwise leaves it unchanged. It is
# described by the Kraus operators
#
# .. math::
#
#     K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-p}\end{pmatrix}, \quad
#     K_1 = \begin{pmatrix}0 & \sqrt{p}\\ 0 & 0\end{pmatrix}.
#
# What damping parameter (:math:`p`) explains the experimental outcome? We can answer this question
# by optimizing the channel parameters to reproduce the experimental
# observation! 💪 Since the parameter :math:`p` is a probability, we use a sigmoid function to
# ensure that the trainable parameters give rise to a valid channel parameter, i.e., a number
# between 0 and 1.
#
ev = 0.9781  # observed expectation value

def sigmoid(x):
    return 1/(1+np.exp(-x))

@qml.qnode(dev)
def damping_circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AmplitudeDamping(sigmoid(x), wires=0)  # p = sigmoid(x)
    qml.AmplitudeDamping(sigmoid(x), wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

######################################################################
# We optimize the circuit with respect to a simple cost function that attains its minimum when
# the output of the QNode is equal to the experimental value:


def cost(x, target):
    return (damping_circuit(x) - target)**2

######################################################################
# All that remains is to optimize the parameter. We use a straightforward gradient descent
# method.

steps = 35

gd = jaxopt.GradientDescent(cost, maxiter=steps)

x = np.array(0.001)
res = gd.run(x, ev)


print(f"QNode output after optimization = {damping_circuit(res.params):.4f}")
print(f"Experimental expectation value = {ev}")
print(f"Optimized noise parameter p = {sigmoid(x.take(0)):.4f}")


print('\nres:',res)

print('\n x.take(0)',x.take(0))

print('zz res.params:',res.params)

print(' converged:',res.converged)
