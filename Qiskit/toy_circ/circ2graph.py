#!/usr/bin/env python3
# https://qiskit.org/documentation/api/qiskit.dagcircuit.DAGCircuit.html

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from pprint import pprint
from qiskit.converters import circuit_to_dag, dag_to_circuit

#...!...!..................
def show_dag_layers(qdag):
  print('\nShow layers of the graph, [gate+qubits(s)+param(s);]')
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
    print('\n%d layer: '%k,end='')
    for op in lay['graph'].op_nodes():
      qL=[qr._index for qr in  op.qargs]
      #print('op:',dir(op))
      numPar=len(op.op._params) # non-zero for U1,U2,U3
      print('  ',op.name, 'q'+str(qL), 'np%d;'%numPar, end='')
  print('\nend-show')

  
#...!...!..................  
def layers_2_gateAddress(qdag):
  print('\n Address uniquely each gate based on layer and qubits')
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
    gateL=[]
    for op in lay['graph'].op_nodes():
      adrL=[str(k),op.name] +[str(qr._index ) for qr in  op.qargs]
      adr='.'.join(adrL)
      gateL.append(adr)
    print(k, ' layer: ',', '.join(gateL))

    
#...!...!..................    
def layers_2_dict(qdag):
  print('\n Order gates chronologically  (per layer) as list of lists')
  outLL=[]
  dagLay=qdag.layers()
  for k,lay in enumerate(dagLay):
    gateL=[]
    for op in lay['graph'].op_nodes():
      one={'name':op.name}
      one['qubit']= [qr._index  for qr in  op.qargs]
      one['param']=op.op._params
      gateL.append(one)
    outLL.append(gateL)
  #pprint(outLL)
  import ruamel.yaml as yaml
  print(yaml.dump(outLL))

  
#...!...!..................
def one_gate_show(qdag,layId,nodeId):
   print('\none_gate_show: layId=%d, nodeId=%d'%(layId,nodeId))
   layIter=qdag.layers()
   from itertools import islice, count
   layer=next(islice(layIter, layId, None))
   nodeIter=layer['graph'].op_nodes()
   node=next(islice(nodeIter,nodeId,None))
   qL=[qr._index for qr in  node.qargs]
   print('node:',node.name, 'q'+str(qL))
   #print(dir(node))
   #print('node._op:',dir(node._op))
   print('params:',node.op._params)
   
#=================================
#=================================
#  M A I N 
#=================================
#=================================

# Create single-qubit gate circuit
qc1 = QuantumCircuit(4)

qc1.s(1)
qc1.u(1.1,2.2,3.3,qubit=0)
qc1.cx(1,2)
qc1.x(1)
qc1.barrier([0, 1,2])
qc1.z(0)
qc1.p(4.4,qubit=1)
qc1.cx(1,0)
qc1.h(2)
qc1.x(3)

print('INPUT qc=')
print(qc1)

qdag = circuit_to_dag(qc1)
#qdag.draw()  # pop-up persistent ImagViewer  (not from Matplotlib)
print('\nList DAG  properties:')
pprint(qdag.properties())

dagLay=qdag.layers()
show_dag_layers(qdag)

one_gate_show(qdag,0, 1)

layers_2_gateAddress(qdag)

layers_2_dict(qdag)

print('\nList of DAG wires (Register,idx), aka qubit lines')
print(qdag.wires,'\n num qubits:',len(qdag.wires))


print('\nconvert DAG back to circ')
circ2=dag_to_circuit(qdag)
print(circ2)

gateL=qdag.op_nodes()  # will include bariers
print('\nOp_nodes: ',len(gateL),'dump first 3' )
myNode=None
for idx,node in enumerate(gateL):
    print(idx,"node name: ", node.name," condition: ", node.condition)
    print("  op: ", node.op)
    print("  qargs: ", node.qargs)
    print("  cargs: ", node.cargs)
    if idx>=2: break


print('\nbuild miniDag with last node:',node.name, type(node))
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister

miniDag = DAGCircuit()
q = QuantumRegister(1, "q")
miniDag.add_qreg(q)
miniDag.apply_operation_front(node.op, qargs=[q[0]], cargs=[])

#miniDag_ch.apply_operation_back(CHGate(), qargs=[q[0],q[1]])
print('this is miniDag:'); print(dag_to_circuit(miniDag))

