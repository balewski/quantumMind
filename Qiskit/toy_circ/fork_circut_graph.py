#!/usr/bin/env python3

# Inserte user gate in layer 2 (in front of h(1) )
# see example at : http://research.classcat.com/quantum/category/qiskit/page/6/

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from pprint import pprint
from qiskit.converters import circuit_to_dag, dag_to_circuit
#from qiskit.extensions.standard import  CHGate, U2Gate
from qiskit.circuit.library import  CHGate, U2Gate
from qiskit.dagcircuit import DAGCircuit


#...!...!....................
def get_names_all_gates(dag):
  print('\n traverse master dag for op nodes, use dag.multigraph_layers()')
  graph_layers = dag.multigraph_layers()  
  outL=[]
  for kLy,graph_layer in enumerate(graph_layers):
    op_nodes = [node for node in graph_layer if node.type == "op"]
    # sort gates according to 1st used qubit
    op_nodes.sort(key=lambda nd: nd.qargs[0].index) 
    
    print(kLy,'layer has num op:',len(op_nodes))
    for kNo, node in enumerate(op_nodes):
      if node.name=='barrier' : continue
      adrL=[str(kLy),node.name] +[str(qr.index) for qr in  node.qargs]
      adr='.'.join(adrL)
      #print(node,adr,node.qargs,type(node))
      #print(kLy,kNo,'-->',adr,node,type(node))
      outL.append(adr)
  return outL

#...!...!....................
def get_gate_by_name(gateName,dag):
  #print('\n traverse master dag for speciffic nodes, use dag.multigraph_layers()')
  graph_layers = dag.multigraph_layers()
  print('search dag for gate:',gateName)
  for kLy,graph_layer in enumerate(graph_layers):
    op_nodes = [node for node in graph_layer if node.type == "op"]
    # sort gates according to 1st used qubit
    op_nodes.sort(key=lambda nd: nd.qargs[0].index)    
    #print(kLy,'layer has num op:',len(op_nodes))
    for kNo, node in enumerate(op_nodes):
      if node.name=='barrier' : continue
      adrL=[str(kLy),node.name] +[str(qr.index) for qr in  node.qargs]
      adr='.'.join(adrL)
      if adr==gateName : return node
  return None

# def remove_op_node(self, node):

#=================================
#=================================
#  M A I N
#=================================
#=================================
# Create the base circuit
qc1 = QuantumCircuit(2)
qc1.x(1)
qc1.s(0)
qc1.cx(1,0)
qc1.h(1)
qc1.barrier([0, 1])
qc1.z(0)
qc1.t(1)
qc1.cx(1,0)

qcBase=qc1
print('qcBase=')
print(qcBase)

baseDag = circuit_to_dag(qcBase)
baseDag.draw()  # pop-up persistent ImagViewer  (not from Matplotlib)

gateNameL=get_names_all_gates(baseDag)
print(len(gateNameL),', gateNameL',gateNameL)

gateNm=gateNameL[3]
print('unit test of finding a gate',gateNm)
node=get_gate_by_name(gateNm,baseDag)
print("node name: ", node.name,node)
print("node qargs: ", node.qargs)

print('\n forking graph, action: replace 1-q gate by U2, cx by ch')

miniDag_u2 = DAGCircuit()
p = QuantumRegister(1, "p")
miniDag_u2.add_qreg(p)
miniDag_u2.apply_operation_back(U2Gate(0.1, 0.2), qargs=[p[0]])
print('this is miniDag_u2:'); print(dag_to_circuit(miniDag_u2))
miniDag_ch = DAGCircuit()
q = QuantumRegister(2, "q")
miniDag_ch.add_qreg(q)
miniDag_ch.apply_operation_back(CHGate(), qargs=[q[0],q[1]])
print('this is miniDag_ch:'); print(dag_to_circuit(miniDag_ch))


print('\n THE fork begins, loop over:',gateNameL)

for kCi,nodeNm in enumerate(gateNameL):
  print('\n',kCi,' circ, change gate=',nodeNm)
  dag1= circuit_to_dag(qcBase) # must create new DAG for in-place changes, IMPORTANT
  node=get_gate_by_name(nodeNm,dag1)
  assert node!=None
  #print("node name: ", node.name,node)
  #print("node qargs: ", node.qargs)
  if len(node.qargs)==1:
    dag1.substitute_node_with_dag(node=node, input_dag=miniDag_u2, wires=[p[0]])
  elif len(node.qargs)==2:
    dag1.substitute_node_with_dag(node=node, input_dag=miniDag_ch, wires=[q[0],q[1]])
  else:
    badCase11
  print(dag_to_circuit(dag1))
  #break
