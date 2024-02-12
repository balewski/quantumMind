#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
explore transpiler options

https://cqcl.github.io/pytket/manual/manual_compiler.html


'''

from access_qtuum import activate_qtuum_api
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit


from pytket import Circuit, OpType
from pytket.predicates import GateSetPredicate, NoMidMeasurePredicate
from pytket.passes import RebaseTket
from pytket.passes import auto_squash_pass

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
    

    print('define some circ')
    qc = Circuit(2, 2)
    qc.Rx(0.2, 0).Ry(0.3, 0).CX(0, 1).Rz(-0.7, 1).measure_all()
    print(tk_to_qiskit(qc))
    print('same as commands:\n',qc.get_commands())
    print('\nM:commands for same circ transpiled for',backend.backend_info.name)
    # https://cqcl.github.io/pytket/manual/manual_compiler.html
    qcT=backend.get_compiled_circuit(qc)
    print(qcT.get_commands())


    #..... Each Predicate can be constructed on its own to impose tests on Circuits during construction.
    gateset = GateSetPredicate({OpType.Rx, OpType.CX, OpType.Rz, OpType.Measure})
    midmeasure = NoMidMeasurePredicate()

    print('M:gateset predicate:',gateset.verify(qc))
    print('M:midmeasure predicate:',midmeasure.verify(qc))

    #.... Rebases (aka change base gates using naive substitution)
    print('\nM:standard rebase pass that converts circ to CX and TK1 gates')
    RebaseTket().apply(qc) 
    print(qc.get_commands())
 
    print('\nM2: apply auto squash into bases defined by user')
    qc2 = Circuit(1).H(0).Ry(0.5, 0).Rx(-0.5, 0).Rz(1.5, 0).Ry(0.5, 0).H(0)
    print(tk_to_qiskit(qc2))
    gates = {OpType.PhasedX, OpType.Rz, OpType.Rx, OpType.Ry,OpType.H}
    custom_squash = auto_squash_pass(gates)
    custom_squash.apply(qc2)
    print('same after a custom basis auto_squash(), here use Qiskit basis')
    print(tk_to_qiskit(qc2))

    
