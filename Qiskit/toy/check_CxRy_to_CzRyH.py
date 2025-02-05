#!/usr/bin/env python3
# test Ry--> X90-Z(phi)-X90  transformation

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RYGate

# Set NumPy print options to display full matrices with 3 decimal places
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2)

def are_unitaries_equal_up_to_phase(U1, U2):
    max_index = np.unravel_index(np.argmax(np.abs(U1)), U1.shape)
    
    if np.abs(U1[max_index]) < 1e-10 or np.abs(U2[max_index]) < 1e-10:
        return np.allclose(U1, U2)
    phase = U1[max_index] / U2[max_index]
    print('relative phase=', phase)
    return np.allclose(U1, U2 * phase, atol=1e-8)

def create_circuitA(th1, th2):  # QCrank 1+1  using CX
    qc = QuantumCircuit(2)
    #qc.h(1) # address 
    #qc.ry(th1, 0)
    qc.cx(0, 1)
    qc.ry(th2, 1)
    qc.cx(0, 1)    
    return qc

def create_circuitB(th1, th2):  # QCrank 1+1  using CX
    qc = QuantumCircuit(2)
    #qc.h(1) # address 
    #qc.ry(th1, 0); qc.h(0)
    qc.h(1)
    qc.cz(0, 1)

    qc.ry(-th2, 1) 
    qc.cz(0, 1)  ; qc.h(1)
    return qc


def check_unitaries(th1, th2,i):
    qcA = create_circuitA(th1, th2)
    qcB = create_circuitB(th1, th2)

    Ua=Operator(qcA).data
    Ub=Operator(qcB).data

    
    print("Ua unitary:\n",Ua)
    print("Ub unitary:\n",Ub)
    
    if 1:
        print(qcA)
        print(qcB)

    are_equal_transformed = are_unitaries_equal_up_to_phase(Ua,Ub)
   
    print(f"\nOriginal and transformed unitaries are equal up to a global phase: {are_equal_transformed}")
    

    return are_equal_transformed 


#=================================
#  M A I N 
#=================================

def main():
    np.random.seed(43)  # for reproducibility
    m = 4  # number of random angle pairs to test
    
    for i in range(m):
        th1 = np.random.uniform(0, 2*np.pi)
        th2 = np.random.uniform(0, 2*np.pi)
        
        print(f"\nTest {i+1}: Ry(θ1={th1:.3f}) and Ry(θ2={th2:.3f})")
        if check_unitaries(th1, th2,i):
            print("All decompositions are correct!")
        else:
            print("Decomposition mismatch detected.")
            exit(0)
        print("-" * 50)
    print('PASS for %d pairs'%m)

if __name__ == "__main__":
    main()
