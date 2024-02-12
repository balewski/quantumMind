__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import scipy.stats
from pprint import pprint

from qiskit.quantum_info.operators import Operator
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumCircuit, QuantumRegister


#...!...!....................
# operator rotating 1 qubit by an angle N(0,theta_std) around random 3D axis
def one_qubit_noisy_rot(theta_std):
    # 4 base 1-qubit opertaors
    e1q_op= Operator([[ 1,0],[0,1]])
    x1q_op= Operator([[ 0,1],[1,0]])
    y1q_op= Operator([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]])
    z1q_op= Operator([[1,0],[0,-1]])

    # rotation angle:
    theta = scipy.stats.truncnorm.rvs( -3, 3 ) * theta_std

    #direction of rotation for 1 qubit
    vec  = np.random.randn( 3 )
    vec /= np.linalg.norm ( vec )
    dir_op= vec[0]* x1q_op +  vec[1]* y1q_op +  vec[2]* z1q_op
    op=  np.cos(theta/2) *e1q_op - (0+1j)* np.sin(theta/2)*dir_op
    return op,float(theta)

# warn: THE NEXT 2 FUNCTIONS MUST BE CHANGED IN SYNC
#...!...!....................
def get_names_all_gates(dag, verb=1):
    if verb>0:
        print('\n traverse master dag for op nodes, use dag.multigraph_layers()')
    graph_layers = dag.multigraph_layers()  
    outL=[]
    for kLy,graph_layer in enumerate(graph_layers):
        op_nodes = [node for node in graph_layer if node.type == "op"]
        # sort gates according to 1st used qubit
        op_nodes.sort(key=lambda nd: nd.qargs[0].index) 
        if verb>0:
            print(kLy,'layer has num op:',len(op_nodes))
        for kNo, node in enumerate(op_nodes):
            if node.name=='barrier' : continue
            if node.name=='measure' : continue
            adrL=['%d'%kLy,node.name] +[str(qr.index) for qr in  node.qargs]
            adr='.'.join(adrL)
            outL.append(adr)
    return outL

#...!...!....................
def get_gate_by_name(gateName,dag,verb=1):
  graph_layers = dag.multigraph_layers()
  if verb>0:  print('search dag for gate:',gateName)
  for kLy,graph_layer in enumerate(graph_layers):
    op_nodes = [node for node in graph_layer if node.type == "op"]
    # sort gates according to 1st used qubit
    op_nodes.sort(key=lambda nd: nd.qargs[0].index)    
    for kNo, node in enumerate(op_nodes):
      if node.name=='barrier' : continue
      if node.name=='measure' : continue
      adrL=['%d'%kLy,node.name] +[str(qr.index) for qr in  node.qargs]
      adr='.'.join(adrL)
      if adr==gateName : return node
  return None

#...!...!....................
def make_noisyMiniDag(node,theta_std, verb=1):
    nqubit=len(node.qargs)
    qL=[x for x in range(nqubit)]
    theta_std/=np.sqrt(len(qL))
    if verb>0:
        print('NoisyMiniDag op=',node.name, 'nqubit=',nqubit,'theat_std=%.3f'%theta_std)
    p = QuantumRegister(nqubit, "p")
    dag= DAGCircuit()
    dag.add_qreg(p)
    
    noise_op,theta=one_qubit_noisy_rot(theta_std)
    qc=dag_to_circuit(dag)
    for q in qL:
        noise_op,theta=one_qubit_noisy_rot(theta_std)
        qc.unitary(noise_op, [q], label='noise_q%d'%q)

    dag=circuit_to_dag(qc)
    
    if nqubit==1:
        dag.apply_operation_back(node.op, qargs=[p[0]], cargs=[])
    else:
        dag.apply_operation_back(node.op, qargs=[p[0],p[1]], cargs=[])
    
    if verb>0:
        print('miniDag'); print(dag_to_circuit(dag))
    return dag,p

#............................
#............................
#............................
class NoisyDagGenerator(object):
    # keeps track of generated circuits
    def __init__(self, baseCirc,verb=1):
        baseDag=circuit_to_dag(baseCirc)
        gateNameL=get_names_all_gates(baseDag, verb=verb)

        if verb>0:
            print('base DAG properties:'); pprint(baseDag.properties())
            print('num 2-q gates',len(baseDag.twoQ_gates()))
            print('numGates=%s, gateNameL'%len(gateNameL),gateNameL)
        self.dag=baseDag
        self.circ=baseCirc
        self.gateNameL=gateNameL
        self.expMeta={}
        
        
    #...!...!....................
    def baseCircuit(self,verb=1):
        circ=dag_to_circuit(self.dag)
        self.expMeta[circ.name]={'injGate': '0.base.0'} # encodes no noise injected
        return circ
    
    #...!...!....................
    def noisyCircuit(self, gateName,theta_std,verb=0):
        if verb>0:
            print('\n ****** XXnoisyCircuitList for gateName=',gateName)
        # must create new DAG for in-place changes, IMPORTANT
        dag1= circuit_to_dag(self.circ) 
        node=get_gate_by_name(gateName,dag1, verb=verb)
        assert node!=None
        if verb>0:
            print("node name: ", node.name,node)
            print("node qargs: ", node.qargs)
        miniDag,p=make_noisyMiniDag(node,theta_std,verb=verb)
        
        if len(node.qargs)==1:
            dag1.substitute_node_with_dag(node=node, input_dag=miniDag, wires=[p[0]])
        elif len(node.qargs)==2:
            dag1.substitute_node_with_dag(node=node, input_dag=miniDag, wires=[p[0],p[1]])
        else:
            badCase11
        circ=dag_to_circuit(dag1)
        circ.name='circ-'+gateName
        self.expMeta[circ.name]={'injGate': gateName}
        return circ
    
