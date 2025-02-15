#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit many circuits  -MORE ***  CAREFULL w/ credits
https://cqcl.github.io/pytket-quantinuum/api/#batching

'''
import time
import json
from pprint import pprint
from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket import Circuit, OpType
from pytket.backends import StatusEnum
from pytket.circuit  import BasisOrder

#...!...!..................
def create_circ( nq):  # TKET circ
    assert nq>0
    qc = Circuit(nq,nq)
    qc.X(nq-1)
    qc.measure_all()
    return qc

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    machine = 'H1-1E'
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine,api_handler=api))
    
    num_circ=3
    print('define %d circuits'%num_circ)
    circL=[] # TKET circ
    for ic in range(num_circ):
        qc=create_circ(1+ic)
        qcT=backend.get_compiled_circuit(qc)
        circL.append( qcT )
        print(tk_to_qiskit(circL[ic]))
    
    assert machine=='H1-1E'
    print("M.... start batching .... ncirc=",num_circ); assert num_circ>0    
    h1 = backend.start_batch(max_batch_cost=1, circuit=circL[0], n_shots=10)
    h2 = backend.add_to_batch(h1, circuit=circL[1], n_shots=15)
    h3 = backend.add_to_batch(h1,  circuit=circL[2], n_shots=20, batch_end=True)       
    jhandle = h3
    jhandL=[h1,h2,h3]  # so I can retrieve all 3
    
    print('submitted %d circ,  %s job_id=%s'%(num_circ,machine,jhandle[0]))

    time.sleep(2)  # give backend time to work

    print('\nRQT: check results for Qtuum job:',jhandle[0])
    t0=time.time()
    while True:  # await for job completion
        try:
            status = backend.circuit_status(jhandle)
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
#    backend.get_results(handle_list)
    for ic in range(3):
        jhand=jhandL[ic]
        result = backend.get_result(jhand)
        #result = backend.get_result(h1)  # reslut for circ 1
    
        # use this to match Qiskit bit order for QCrank
        tket_basis=BasisOrder.dlo # DLO gives (c[1], c[0]) == (1, 0)
        tket_counts=result.get_counts(basis=tket_basis)
        print('\ncirc %d  Tket counts:'%ic); pprint(tket_counts)

    print('M: done one job with 3 circ')
