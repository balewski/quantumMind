#!/usr/bin/env python3

# Inserte user gate in layer 2 (in front of h(1) )
# see example at : http://research.classcat.com/quantum/category/qiskit/page/6/

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from pprint import pprint
from qiskit.converters import circuit_to_dag, dag_to_circuit

def show_dag_layers(qdag):
  print('\nLayers of the graph, only gates are printed')
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
     print(k, ' layer: ',end='')
     for op in lay['graph'].op_nodes():
        qL=[qr.index for qr in  op.qargs]
        print('  ',op.name, 'q'+str(qL))
 

   
# Create single-qubit gate circuit
qc1 = QuantumCircuit(2)
qc1.x(0)
qc1.s(1)
qc1.cx(0,1)
qc1.h(1)
qc1.barrier([0, 1])
qc1.z(0)
qc1.t(1)
qc1.cx(1,0)

print('qc1=')
print(qc1)


qdag = circuit_to_dag(qc1)
#1 qdag.draw()  # pop-up persistent ImagViewer  (not from Matplotlib)
show_dag_layers(qdag)

print('\nAdd the H-gate  after circuit')
from qiskit.extensions.standard import HGate, CnotGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
q = QuantumRegister(2, 'q')
qdag.apply_operation_back(HGate(), qargs=[q[1]])
print(dag_to_circuit(qdag))

print('\nAdd the CX-gate before circuit')
qdag.apply_operation_front(CnotGate(), qargs=[q[1],q[0]], cargs=[])
print(dag_to_circuit(qdag))

print('\nReplace the node with a subcircuit ')
from qiskit.extensions.standard import CHGate, U2Gate, CnotGate
from qiskit.dagcircuit import DAGCircuit
mini_dag = DAGCircuit()
p = QuantumRegister(2, "p")
mini_dag.add_qreg(p)
mini_dag.apply_operation_back(CHGate(), qargs=[p[1], p[0]])
mini_dag.apply_operation_back(U2Gate(0.1, 0.2), qargs=[p[1]])
print('this is mini_dag:'); print(dag_to_circuit(mini_dag))

print(' substitute the cx node with the above mini-dag')
cx_node = qdag.op_nodes(op=CnotGate).pop()
qdag.substitute_node_with_dag(node=cx_node, input_dag=mini_dag, wires=[p[0], p[1]])


qc2=dag_to_circuit(qdag)
print(qc2)
print('Must regenerate dag after adding gates to get natural indexing') 
qdag2=circuit_to_dag(qc2)

gateL=qdag2.gate_nodes()
print('\ngate_nodes: ',len(gateL))
node=gateL[8]
print('Replacing:')
print("node name: ", node.name)
print("node op: ", node.op)
print("node qargs: ", node.qargs)
print("node cargs: ", node.cargs)
print("node condition: ", node.condition)

qdag2.substitute_node_with_dag(node=node, input_dag=mini_dag, wires=[p[0], p[1]])

print(dag_to_circuit(qdag2))

print('\n insert my Unitary=iSWAP in front of cx gate #4')
gateL=qdag2.gate_nodes()
print('\ngate_nodes: ',len(gateL))
node=gateL[4]
print("target node name: ", node.name)
print("node qargs: ", node.qargs)

# assemble mini_dqg from qc
from qiskit.quantum_info.operators import Operator

# iSWAP matrix operator
iswap_op = Operator([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]])
# Using operators in circuits
qc0 = QuantumCircuit(2)
qc0.unitary(iswap_op, [0, 1], label='iswap')
qdag0=circuit_to_dag(qc0)
qdag0.apply_operation_back(node.op, qargs=node.qargs)
print('my mini dag')
print(dag_to_circuit(qdag0))

qdag2.substitute_node_with_dag(node=node, input_dag=qdag0, wires=[q[0], q[1]])
print('final circuit qc3')
qc3=dag_to_circuit(qdag2)
print(qc3)


optLev=1
print('\n Transpile=(Optimize and fuse gates)  qc3--> qc4, optLev=',optLev)
qc4 = transpile(qc3, basis_gates=['u3','cx'], optimization_level=optLev)
print(qc4)



