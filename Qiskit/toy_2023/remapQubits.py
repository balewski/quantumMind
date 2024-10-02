#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Using operators in circuits
example how to remap qubits for a circuit

'''

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import QuantumRegister
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout


#...!...!....................
def remap_qubits(qc,targetMap):
    # access quantum register
    qr=qc.qregs[0]
    nq=len(qr)
    assert len(targetMap)==nq
    print('registers has %d qubist'%nq,qr)
    regMap={}  
    for i,j in enumerate(targetMap):
        #print(i,'do',j)
        regMap[qr[j]]=i
        
    #print('remap qubits:'); print(regMap)
    layout = Layout(regMap)
    #print('layout:'); print(layout)
    
    # Create the PassManager that will allow remapping of certain qubits
    pass_manager = PassManager()
    
    # Create the passes that will remap your circuit to the layout specified above
    set_layout = SetLayout(layout)
    apply_layout = ApplyLayout()

    # Add passes to the PassManager. (order matters, set_layout should be appended first)
    pass_manager.append(set_layout)
    pass_manager.append(apply_layout)
    
    # Execute the passes on your circuit
    remapped_circ = pass_manager.run(qc)
    
    return remapped_circ
    
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    num_qub=6
    # Create circuit with 6 qubits
    qc = QuantumCircuit(num_qub,num_qub)

    # Add gates & meas
    qc.ch(1,3)
    qc.s(1)
    qc.t(0)
    qc.cx(0,1)
    qc.x(0)
    qc.sdg(0)
    qc.s(1)
    qc.barrier([0,1])
    qc.measure([0,1], range(2))
    print(qc)

    qMap=[]
    nq_data=3
    for i in range(num_qub):
        j=i
        if i <nq_data:  j=nq_data-i-1
        qMap.append(j)

    print('M:qmap',qMap)
    qc=remap_qubits(qc,qMap)
    print(qc)
