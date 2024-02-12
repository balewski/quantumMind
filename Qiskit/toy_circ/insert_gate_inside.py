#!/usr/bin/env python3

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from pprint import pprint
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import U3Gate,HGate
from qiskit.dagcircuit import DAGCircuit

# Create single-qubit gate circuit
qc1 = QuantumCircuit(2)
qc1.cx(0,1)
qc1.u3(0.1,0.2,0.3,0)
qc1.x(0)
qc1.s(1)
qc1.h(1)
qc1.cx(0,1)

print('qc1=')
print(qc1)

optLev=1
print('\n Transpile=(Optimize and fuse gates)  qc3--> qc4, optLev=',optLev)
qc2 = transpile(qc1, basis_gates=['u3','cx'], optimization_level=optLev)
print(qc2)
dag = circuit_to_dag(qc2)

print(' insert H before  U3 node with  mini-dag')
gateL=dag.gate_nodes()
print('\ngate_nodes: ',len(gateL))
node=gateL[2]
print("target node name: ", node.name)
print("node qargs: ", node.qargs)

p = QuantumRegister(1, "p")
miniDag= DAGCircuit()
miniDag.add_qreg(p)
miniDag.apply_operation_back(HGate(), qargs=[p[0]])
miniDag.apply_operation_back(node.op, qargs=[p[0]], cargs=[])

print('\nThis is miniDag:'); print(dag_to_circuit(miniDag))

dag.substitute_node_with_dag(node=node, input_dag=miniDag, wires=[p[0]])
print('final circuit qc3')
qc3=dag_to_circuit(dag)
print(qc3)

