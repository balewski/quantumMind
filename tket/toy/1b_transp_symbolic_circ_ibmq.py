#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
explore transpilation of symbolic circuit against IBMQ QPU

https://cqcl.github.io/pytket/manual/manual_compiler.html#compiling-symbolic-circuits

NOT related to QTUUM !

'''

from pytket import Circuit, Qubit, OpType
from pytket.passes import auto_squash_pass
from pytket.extensions.qiskit import AerStateBackend,AerBackend,IBMQEmulatorBackend # ibmq_belem, 5-q
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from sympy import symbols

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.qasm import circuit_from_qasm, circuit_to_qasm_str

#...!...!....................
def make_param_circ(a,b):
    circ = Circuit(4)
    circ.Ry(a, 0)
    circ.Ry(a, 1)
    circ.CX(0, 1)
    circ.Rz(b, 1)
    circ.CX(1, 2)
    circ.CX(2, 3)
    return circ

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    a, b = symbols("a b")
    qc=make_param_circ(a,b)
    print('M: org circ')
    print(tk_to_qiskit(qc))
   
    for x in qc.get_commands()  : print(x)

    # .... work with yields
    if 1: # realistic simulator executed as pytket on IBMQ
        from qiskit import IBMQ

        print('M: IBMQ-load account ...')
        IBMQ.load_account()
        #b_emu = IBMQEmulatorBackend("ibmq_belem", hub="ibm-q", group="open", project="main")
        b_emu = IBMQEmulatorBackend( "ibmq_guadalupe",hub='ibm-q-ornl', group='lbnl', project='chm170')

        print('M:b_emu.backend_info.name:', b_emu.backend_info.name) 
        #print('M:b_emu.backend_info:', b_emu.backend_info)  # large output!

        b_emu.default_compilation_pass(optimisation_level=2)
        qc2 = b_emu.get_compiled_circuit(qc)   # Compile once outside of the objective function

        print('M: transpiled qc2 as QASM')
        print(circuit_to_qasm_str(qc2))
        print(tk_to_qiskit(qc2))
       
        
        aval=0.1; bval=0.2
        print('substitute values',aval,bval)
        qc2.symbol_substitution({a : aval, b : bval})
        print(tk_to_qiskit(qc2))
