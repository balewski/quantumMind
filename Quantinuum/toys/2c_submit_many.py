#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit many circuits  -MORE ***  CAREFULL w/ credits
 w/o using batching (it is less restrictive)
exercise reset(0) gate

'''
import time
import json
from pprint import pprint
from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket.backends import StatusEnum
from pytket.circuit  import BasisOrder

from qiskit import  QuantumCircuit
#...!...!..................
def create_circ( nq):  # Qiskit circ
    assert nq>0
    qc = QuantumCircuit(nq, nq)
    qc.h(0)
    for j in range(1,nq):
        qc.cx(0,j)
    qc.reset(0)
    qc.barrier()
    qc.measure(range(nq), range(nq))
    return qc


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    machine = 'H1-1E'
    #machine = 'H1-1SC'
    
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine,api_handler=api))
    
    num_circ=3
    print('define %d circuits'%num_circ)
    circL=[] # Qiskit circ
    shotL=[]
    for ic in range(num_circ):
        qc=create_circ(1+ic)  # Qiskit circ
        print(qc)
        qcT=backend.get_compiled_circuit(qiskit_to_tk(qc))
        circL.append( qcT )  # TKET circ
        shot=10+5*ic
        shotL.append(shot*20)

    assert machine!='H1-1'
    print("M.... start multiple .... ncirc=",num_circ); assert num_circ>0
    jhandL=[ 0 for i in range(num_circ) ]
    for i in range(num_circ):
        jhandL[i] = backend.process_circuit( circuit=circL[i], n_shots=shotL[i])
       
    jhand=jhandL[-1]
    print('submitted %d circ,  %s job_id=%s'%(num_circ,machine,jhand))

    time.sleep(2)  # give backend time to work


    print('\nRQT: check results for the last Qtuum job:',jhand)
    t0=time.time()
    while True:  # await for job completion
        try:
            status = backend.circuit_status(jhand)
        except:
            print('qtuum job %s NOT found, quit\n'%job_id); exit(99)

        msg=json.loads(status.message)
        if status.status in (StatusEnum.COMPLETED, StatusEnum.ERROR):
            break
        t1=time.time()
        sleep_sec=5
        print('job status:',status.status,'position=',msg["queue-position"],' elaT=%.0d , sleep %d ...'%(t1-t0,sleep_sec))
        if  status.status ==StatusEnum.CANCELLED: exit(0)
        time.sleep(sleep_sec)  # give backend time to work

    #print(status)
    pprint(msg)

    print('M: access results ....')
    resultL=backend.get_results(jhandL)
    for ic in range(num_circ):
        result = backend.get_result(jhandL[ic])
    
        # use this to match Qiskit bit order for QCrank
        tket_basis=BasisOrder.dlo # DLO gives (c[1], c[0]) == (1, 0)
        tket_counts=result.get_counts(basis=tket_basis)
        print('\ncirc %d  Tket counts:'%ic); pprint(tket_counts)

    print('M: done one job with all circ')
