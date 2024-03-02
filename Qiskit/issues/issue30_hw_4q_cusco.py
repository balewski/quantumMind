#!/usr/bin/env python3
#  computs fead-foorward condition base don hamming weight
#  works  on Aer for any niq, hwSel.
# crashes on HW for niq>=5


import numpy as np
from qiskit.circuit.classical import expr
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService
from time import time, sleep
from qiskit.providers.jobstatus import JobStatus

from itertools import combinations

#...!...!....................
def feedF_ni_qInp_hamming_weight(niq, hwSel):
    assert 0 <= hwSel <= niq
    qa = niq  # Index of the ancilla qubit
    qr = QuantumRegister(niq + 1)  # Quantum register for input qubits + 1 auxiliary qubit
    cr = ClassicalRegister(niq)  # Classical register for measuring input qubits
    cro = ClassicalRegister(1)  # Output classical register
    qc = QuantumCircuit(qr, cr, cro)

    # Prepare the input qubits in superposition states
    for i in range(niq):
        qc.h(i)
    qc.barrier()

    # Measure the input qubits
    for i in range(niq):
        qc.measure(i, cr[i])
    qc.barrier()

    # Compute conditions for various Hamming weights dynamically
    hw_conditions = []
    for hw in range(niq + 1):
        if hw == 0 or hw == niq:
            # Special cases: HW=0 (all qubits are 0) and HW=niq (all qubits are 1)
            condition = expr.logic_not(cr[0]) if hw == 0 else cr[0]
            for i in range(1, niq):
                bit_condition = expr.logic_not(cr[i]) if hw == 0 else cr[i]
                condition = expr.bit_and(condition, bit_condition)
        else:
            # General cases: 1 <= HW <= niq-1
            condition_parts = []
            for bits in combinations(range(niq), hw):
                part = cr[bits[0]]
                for bit in bits[1:]:
                    part = expr.bit_and(part, cr[bit])
                # Add conditions for bits not in the combination to be 0
                for not_bit in set(range(niq)) - set(bits):
                    part = expr.bit_and(part, expr.logic_not(cr[not_bit]))
                condition_parts.append(part)
            condition = condition_parts[0]
            for part in condition_parts[1:]:
                condition = expr.bit_or(condition, part)
        hw_conditions.append(condition)

    # Conditional execution based on the selected Hamming weight
    if hw_conditions[hwSel] is not None:
        with qc.if_test(hw_conditions[hwSel]):
            qc.x(qr[qa])  # Apply X gate to the auxiliary qubit
    
    # Unconditionally measure the auxiliary qubit
    qc.measure(qr[qa], cro[0])

    return qc

 #...!...!....................
from scipy.special import comb
def n_choose_k(n, k):
    return comb(n, k)

#...!...!....................
def ana_counts(countsIn, niq, hwSel,cThr=50):
   
    # Filter dictionary by value
    counts = {key: value for key, value in countsIn.items() if value >= cThr}
    niqnp=len(countsIn)
    nDrop=niqnp - len(counts)
    print('\nANA: nDrop=%d  of %d, cThr=%d'%(nDrop, niqnp,cThr))
    # Split the dictionary based on the key starting with '0 ' or '1 '
    counts_0 = {key: value for key, value in counts.items() if key.startswith('0 ')}
    counts_1 = {key: value for key, value in counts.items() if key.startswith('1 ')}
    
    print("%d keys with '0 ': "%(len(counts_0)),counts_0)
    print("%d keys with '1 ': "%(len(counts_1)),counts_1)


    n1k=n_choose_k(niq,hwSel)
    print('expected num_1:%d  niq=%d  hwSel=%d'%(n1k,niq,hwSel))
    assert n1k==len(counts_1)
    print('--- PASS ---')
       
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    niQ=3   # select number of input qubits
    hwSel=2  # select desired HW here

    # works for: niQ=4, but crashes on HW  for niQ=5
    
    qc=feedF_ni_qInp_hamming_weight(niq=niQ, hwSel=hwSel)
    
    print(circuit_drawer(qc, output='text',cregbundle=False, idle_wires=False))
    # (qc.decompose()
    shots=4000

    if 0:  #do HW
        print('M: access QiskitRuntimeService()...')
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backN="ibm_cusco"
        backend = service.get_backend(backN)
        if "if_else" not in backend.target:
            from qiskit.circuit import IfElseOp
            backend.target.add_instruction(IfElseOp, name="if_else")
        print('M: transpiling for ...',backend)
        qcT = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
        physQubitLayout = qcT._layout.final_index_layout(filter_ancillas=True)
        print('M: executing on qubits',physQubitLayout)
        job=backend.run(qcT,shots=shots, dynamic=True)
    else:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        job =  backend.run(qc,shots=shots)

    print('use backend=%s version:'%backend,backend.version )
   
     
    i=0; T0=time()
    while True:
        jstat=job.status()
        elaT=time()-T0
        print('P:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
        if jstat in [ JobStatus.DONE, JobStatus.ERROR]: break
        i+=1; sleep(5)

    print('M: job done, status:',jstat,backend)

    result = job.result()
    counts=result.get_counts(0)
    print('M: counts:', counts)

    ana_counts(counts,niQ,hwSel, cThr=50)
