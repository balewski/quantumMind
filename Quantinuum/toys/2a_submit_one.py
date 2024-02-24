#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit 1 circuit  - CAREFULL w/ credits

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
def create_circ( N=1):  # TKET circ
    qc = Circuit(1,1)
    for i in range(N):
        qc.X(0)
    qc.H(0)    
    qc.measure_all()
    return qc

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    machine = 'H1-1SC'  # cost estimator
    machine = 'H1-1E'  # emulator
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine,api_handler=api))
    

    print('define some TKET circ')
    qc=create_circ(3)
    print(tk_to_qiskit(qc))

    print('commands for same circ transpiled for',machine)
    qcT=backend.get_compiled_circuit(qc)

    for x in qcT.get_commands() :  print(x)

    assert machine!='H1-1'  #real HW
    assert machine not in ['H1-1','H2-1']  #real HW
     
    jhandle = backend.process_circuit(qcT, n_shots= 10)
  
    print('submitted qtuum %s job_id: %s'%(machine,jhandle[0]))

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
    result = backend.get_result(jhandle)
    # use this to match Qiskit bit order for QCrank
    tket_basis=BasisOrder.dlo # DLO gives (c[1], c[0]) == (1, 0)
    tket_counts=result.get_counts(basis=tket_basis)
    print('\nQCrank  Tket counts:'); pprint(tket_counts)
    
    print('M: done one job')
