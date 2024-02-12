#!/usr/bin/env python3
# from https://pyquil-docs.rigetti.com/en/stable/start.html#getting-started
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quilbase import Declare

from pyquil import get_qc, Program
from pyquil.gates import CNOT, Z, MEASURE
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare

prog = Program(
    Declare("ro", "BIT", 2),
    Z(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(10)

#with local_forest_runtime():
#    qvm = get_qc('9q-square-qvm')
#    bitstrings = qvm.run(qvm.compile(prog)).readout_data.get("ro")

# Get a quantum virtual machine (simulator)
qvm = get_qc("2q-qvm")
# Execute the program synchronously
res=qvm.run(prog).readout_data.get("ro")
print('M:res1',res)
