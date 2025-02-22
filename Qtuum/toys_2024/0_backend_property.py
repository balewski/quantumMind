#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
explore  backend properties

https://cqcl.github.io/pytket/manual/manual_compiler.html


'''

from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit


from pytket import Circuit, OpType
from pytket.predicates import GateSetPredicate, NoMidMeasurePredicate
from pytket.passes import RebaseTket
from pytket.passes import auto_squash_pass

from pprint import pprint
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


    x=backend.backend_info.architecture

    print('back arch:',x)
    print(backend.backend_info)
