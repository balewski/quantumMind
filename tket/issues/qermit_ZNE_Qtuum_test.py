#!/usr/bin/env python3
'''
Collect ZNE data 
Test in Qtuum 

'''



from pytket.extensions.quantinuum import QuantinuumBackend
from pytket import Circuit, OpType

from pytket.extensions.qiskit import tk_to_qiskit
import qiskit as qk
from pprint import pprint
from qermit.zero_noise_extrapolation import Folding
import numpy as np

folding_type=Folding.two_qubit_gate


#...!...!....................
def build_circuit_mcx(nq = 5):
    circuit = Circuit(n_qubits=nq)

    for i in range(nq-1): circuit.X(i)
    circuit.add_gate(OpType.CnX, [j for j in range(nq)])
    
    return circuit


#...!...!....................
def add_measurements(c):
    nq=c.n_qubits
    c.add_c_register(name='c', size=nq)
    qL=[ c.qubits[j] for j in range(nq)]
    c.add_barrier(qL)
    for j in range(nq):
        c.Measure(c.qubits[j],c.bits[j])

        
#...!...!....................
def print_qc(qc,text='myCirc'):
    cxDepth=qc.depth_by_type(OpType.CX)
    print('\n---- %s ---- CX depth=%d'%(text,cxDepth))
    print(tk_to_qiskit(qc))

#...!...!....................
def run_measurement(qc0):
    qc=qc0.copy()
    add_measurements(qc)    
    print('simu  shots=%d'%(shots),tk_backend) 
    handle = tk_backend.process_circuit(qc, n_shots=shots)
    counts = tk_backend.get_result(handle).get_counts()  
    print('M:counts',counts)
    return counts
    

#...!...!....................
def activate_qtuum_api():
    from pytket.extensions.quantinuum.backends.credential_storage import  MemoryCredentialStorage
    from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
    import os

    MY_QTUUM_NAME=os.environ.get('MY_QTUUM_NAME')
    MY_QTUUM_PASS=os.environ.get('MY_QTUUM_PASS')
    print('credentials MY_QTUUM_NAME=',MY_QTUUM_NAME)
    cred_storage = MemoryCredentialStorage()
    cred_storage._save_login_credential(user_name=MY_QTUUM_NAME, password=MY_QTUUM_PASS)
    api = QuantinuumAPI(token_store = cred_storage)
    return api

#=================================
#  M A I N 
#=================================

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    shots=2000
    nRepeat=2
    optL=2
   
    qcM=build_circuit_mcx(6)  
    nq=qcM.n_qubits
    
    print_qc(qcM,'master circuit, %d shots'%shots)


    machine = 'H1-1E'
    api=activate_qtuum_api()
    tk_backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", tk_backend.device_state(device_name=machine, api_handler=api))
    
    qcMT = tk_backend.get_compiled_circuit(circuit=qcM, optimisation_level=optL)
    print_qc(qcMT,'master transpiled')
    
    outL=[]
    noise_scale=1.6
    key0=(1,)*nq
    for i in range(5):
        noise_scale=1. + i*0.5
        for j in range(nRepeat):
            qcFN = folding_type(noise_scaling=noise_scale, circ=qcMT)
            qqDepth=qcFN.depth_by_type(OpType.ZZPhase)            
            print_qc(qcFN,'%d folded, noise scale=%.1f'%(i,noise_scale))
            print('run i=%d j=%d noise=%.1f  qqdDepth=%d'%(i,j,noise_scale,qqDepth))
            counts=run_measurement(qcFN)
            outL.append([noise_scale,qqDepth,counts[key0]])
        
    print('M: outL',outL)
