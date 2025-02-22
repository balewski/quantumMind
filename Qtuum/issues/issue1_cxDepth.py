#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.circuit import OpType
from pytket import Circuit
from pytket.passes import RebaseTket

#...!...!..................
def create_wide_circ(nqa, N=1):  # TKET circ
    M=N+1
    print('define %d-q wide circ'%(M*nqa))
    qc = Circuit(M*nqa, M*nqa)
    for i in range(nqa):
        for j in range(N):
            qc.CX(M*i,M*i+1+j)
    return qc



#=================================
#  M A I N
#=================================
if __name__ == "__main__":

    machine = 'H1-1E'
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine, api_handler=api))

    nbl=6
    qc=create_wide_circ(nbl,3) # TKET circ
    qc1=tk_to_qiskit(qc)
    print(qc1)
    print('ideal tket qc depth:',qc.depth_by_type({ OpType.CX}))
    
    qcT=backend.get_compiled_circuit(qc)
    print('transp tket qc depth:',qcT.depth_by_type({OpType.ZZPhase, OpType.CX}))
    #print(qcT.get_commands())
    
    qcT1=tk_to_qiskit(qcT)
    print(qcT1)


    
    
