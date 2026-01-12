#!/usr/bin/env python3
'''
Tutoral for computing expectation value of an operator
Using shots or matrix  using Qiskit Aqua 

My notes
https://docs.google.com/document/d/1I0ZamueYQCOqZTjGY4ml1dmdNpnEs6SOPTgkvw3GJ6o/edit?usp=sharing

Notation: psi - state, op - operator
Expectation value E = <psi|op|psi>

Based on
https://quantumcomputing.stackexchange.com/questions/12080/evaluating-expectation-values-of-operators-in-qiskit

'''
import numpy as np
from pprint import pprint
from qiskit import QuantumCircuit
from qiskit.aqua.operators  import CircuitOp,CircuitStateFn

# for backend & shots
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn

import qiskit
qiskit.__qiskit_version__


#...!...!....................
def OpZZ():
    circ = QuantumCircuit(2)
    circ.z(0)
    circ.z(1)
    op = CircuitOp(circ)  # and convert to an operator
    return op

#...!...!....................
def OpH2molecule():
    from qiskit.aqua.operators import X, Y, Z, I
    op =  (-1.0523732 * I^I) + (0.39793742 * I^Z) + (-0.3979374 * Z^I) \
    + (-0.0112801 * Z^Z) + (0.18093119 * X^X)
    return op

#...!...!....................
def Psi(x0,x1):  # initial state as 2-bit string
    psi = QuantumCircuit(2)
    if x0: psi.x(0)
    if x1: psi.x(1)
    #psi.sx(0)
    psi = CircuitStateFn(psi) # convert to a state
    return psi
    

#=================================
#=================================
#  M A I N
#=================================
#=================================

print('\n ***** 1 *****  E=<psi|op|psi> using matrix')
#1) define your operator as CircuitOp

op=OpZZ()
print('op: operator is like a circuit'); print(op)

#2)  define the state you w.r.t. which you want the expectation value
psi=Psi(1,1)
print('psi:'); print(psi)

#3 # easy expectation value, use for small systems only!
print('Math1:', psi.adjoint().compose(op).compose(psi).eval().real)
# more elegant
expectation_value = (~psi @ op @ psi).eval()
print('Math2: C=', expectation_value, 'Re=',expectation_value.real)

print('\n ***** 2 *****  E=<psi|op|psi> using shots, Op=H2 molecule')
op=OpH2molecule()
print('op: H2-molecule Hamiltonian'); print(op)
E_op=(~psi @ op @ psi).eval().real
print('Math2: E_op=',E_op)

# now compute the same using shots

# define your backend or quantum instance (where you can add settings)
backend = Aer.get_backend('qasm_simulator') 
q_instance = QuantumInstance(backend, shots=32)

# define the state to sample
measurable_expression = StateFn(op, is_measurement=True).compose(psi) 

# convert to expectation value
expectation = PauliExpectation().convert(measurable_expression)  


# evaluate
for i in range(3):
    print('\n\n = = = = = =  run %d circuit sampler = = = = = = = = '%i)
        
    # get state sampler (you can also pass the backend directly)
    sampler = CircuitSampler(q_instance).convert(expectation) 
    print('sampler'); print(sampler)
    print(i,'Sampled:', sampler.eval().real)  
