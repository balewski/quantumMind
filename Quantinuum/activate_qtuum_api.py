#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
testing access to Quantinuum API

Yao: QuantinuumBackend uses a memory based storage class (MemoryCredentialStorage) as the default credential storage.

*) for better security, you should retrieve your username and password from a secured storage.
*) available_devices() and and device_state() are the only two methods that require special handling because they are class methods. 

'''

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket import Circuit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit


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
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    machine = 'H1-1E'
    api=activate_qtuum_api()
    backend = QuantinuumBackend(device_name=machine, api_handler=api)
    print('machine=',machine)
    print("status:", backend.device_state(device_name=machine, api_handler=api))
    print([x.device_name for x in backend.available_devices(api_handler=api)])

    print('define Bell-state circ')
    qc = Circuit(2).H(0).CX(0,1).measure_all()
    print(tk_to_qiskit(qc))
    print('same as commands:\n',qc.get_commands())

    print('commands for same circ transpiled for',machine)
    # https://cqcl.github.io/pytket/manual/manual_compiler.html
    qcT=backend.get_compiled_circuit(qc)
    print(qcT.get_commands())
    # the map from bit to position in the measured state
    print('M:bit_readout',qcT.bit_readout)

    # the map from qubit to position in the measured state
    print('M:qubit_readout',qcT.qubit_readout)
        
    # the map from qubits to the bits to which their measurement values were written
    print('M:qubit_to_bit_map',qcT.qubit_to_bit_map)
    print('M: circuit NOT executed ')
    
    exit(0)
    this_would_cost_you
    # see toys$ ./2a_submit_one.py 
    jhandle = backend.process_circuit(qcT, n_shots= 10)
    print('submitted qtuum %s job_id: %s'%(machine,jhandle[0]))
    




