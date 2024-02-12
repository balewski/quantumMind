#!/usr/bin/env python3
# https://github.com/CQCL/pytket/tree/main/examples

print('M:start...')

# ------ Pyket  circuit -----
from pytket import Circuit
c = Circuit(2,2) # define a circuit with 2 qubits and 2 bits
c.H(0)           # add a Hadamard gate to qubit 0
c.Rz(0.25, 0)    # add an Rz gate of angle 0.25*pi to qubit 0
c.CX(1,0)        # add a CX gate with control qubit 1 and target qubit 0
c.measure_all()  # measure qubits 0 and 1, recording the results in bits 0 and 1

# ------ run on Qiksit Aer simulator
from pytket.extensions.qiskit import AerBackend
b = AerBackend()                            # connect to the backend
compiled = b.get_compiled_circuit(c)        # compile the circuit to satisfy the backend's requirements
handle = b.process_circuit(compiled, 100)   # submit the job to run the circuit 100 times
counts = b.get_result(handle).get_counts()  # retrieve and summarise the results
print(counts)

# expected output: {(0, 0): 49, (1, 0): 51}

# ----- more operations on the circuit ----
print('nqub',c.n_qubits,c.name,c.depth())
from pytket.circuit import OpType
print('depth:',c.depth_by_type(OpType.CU1), c.depth_by_type(OpType.H))

# ---- visualisation ---
from pytket.utils import Graph
G = Graph(c)
G.get_DAG()
G.get_qubit_graph()

# via Qiskit
from pytket.extensions.qiskit import tk_to_qiskit
print(tk_to_qiskit(c))

# via Cirq
from pytket.extensions.cirq import tk_to_cirq
print(tk_to_cirq(c))


# via latex
c.to_latex_file("c.tex")


# ----- commands
cmds = c.get_commands()
print('Commands:\n',cmds)
