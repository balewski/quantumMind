#!/usr/bin/env python3
'''
Collect ZNE data 
Test in Qtuum H1-1E https://docs.google.com/document/d/1xbf2hDqvT4TlvK3-IGt1gJpr9iHzhZArUGSf92mTcng/edit?usp=sharing 

or FakeJakarta  https://docs.google.com/document/d/1osb3mFCww3huKpQK17G9tQaaB_z37wTQvcHtZrvzyYQ/edit?usp=sharing 


'''

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket import Circuit, OpType
from pytket.extensions.qiskit import tk_to_qiskit
from qermit.zero_noise_extrapolation import Folding

from pprint import pprint


#...!...!....................
def build_circuit1():
    qc = Circuit(3) 
    qc.X(0); qc.X(1)
    qc.add_gate(OpType.CnX, [0,1,2])
    #1qc.measure_all()
    return qc

#...!...!....................
def build_circuit_mcx(nq = 5):
    circuit = Circuit(n_qubits=nq,n_bits=nq)
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
   
    shots=2000
    nRepeat=2
    optL=2
   
    qcM=build_circuit_mcx(6)  
    #qcM=build_circuit1()
    nq=qcM.n_qubits
    
    print_qc(qcM,'master circuit, %d shots'%shots)

    if 0:
        from pytket.extensions.qiskit import AerBackend
        tk_backend = AerBackend()
        machine='Aer'
        
    if 1:  # ------ run noisy IBMQ simulator
        from pytket.extensions.qiskit import IBMQEmulatorBackend
        tk_backend = IBMQEmulatorBackend( backend_name='ibmq_jakarta',instance='', )
        machine='FakeJakart'
        
    if 0:
        machine = 'H1-1E'
        api=activate_qtuum_api()
        tk_backend = QuantinuumBackend(device_name=machine, api_handler=api)
        print('machine=',machine)
        print("status:", tk_backend.device_state(device_name=machine, api_handler=api))
   
    qcMT = tk_backend.get_compiled_circuit(circuit=qcM, optimisation_level=optL)
    #1print_qc(qcMT,'master transpiled')
    
    if 0:  run_measurement(qcMT); only_transp_circ

    outL=[]
    noise_scale=1.6
    key0=(1,)*nq
    for i in range(5):
        noise_scale=1. + i*0.5  # for QTUUM
        #noise_scale=1. + i*0.2  # for Jakarta
        for j in range(nRepeat):
            qcFN = Folding.two_qubit_gate(noise_scaling=noise_scale, circ=qcMT)
            if machine =='H1-1E':
                qqDepth=qcFN.depth_by_type(OpType.ZZPhase)
            else:
                qqDepth=qcFN.depth_by_type(OpType.CX)

            #1print_qc(qcFN,'%d folded, noise scale=%.1f'%(i,noise_scale))
            print('run i=%d j=%d noise=%.1f  qqdDepth=%d'%(i,j,noise_scale,qqDepth))
            counts=run_measurement(qcFN)
            outL.append([noise_scale,qqDepth,counts[key0]])
        
    print('M: outL',outL)

