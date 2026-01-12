#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Implements 2-bit Grover algo 
 2-bit oracle as a Toffoli gate and arbitrary 2-bit secret

Here you can get new token:
 https://quantumexperience.ng.bluemix.net/qx/account/advanced
'''

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import Aer, execute

backendSim=Aer.get_backend("qasm_simulator")
backendState=Aer.get_backend("statevector_simulator")
nclbit=2 ; nqubit=3
secret='01'
assert len(secret)==nclbit
print('Exercise Grover search for secret=',secret)

#............................
#............................
def circuit_oracle_2bit(qc,qr,secret):
    #print('CO: secret',secret,' len=',nclbit)
    zerL = [nclbit-i-1 for i, s in enumerate(secret) if '0'==s]
    for i in zerL:  qc.x(qr[i])  # flip 0-s to 1
    qc.ccx(qr[0], qr[1], qr[2])
    for i in zerL:  qc.x(qr[i])  # undo flip

    
#............................
#............................
def circuit_condShift_2bit(qc,qr):
    for i in range(nclbit): qc.x(qr[i])
    qc.h(qr[1])
    qc.cx(qr[0],qr[1])
    qc.h(qr[1])
    #qc.barrier()
    for i in range(nclbit): qc.x(qr[i])


#............................
#............................
def build_oracle_only(xI):
    qr = QuantumRegister(nqubit,'q')
    cr = ClassicalRegister(1,'c') #   oracle output
    qc = QuantumCircuit(qr, cr)
    
    # change input from |0> to |1> as defined by xI
    oneL = [nclbit-i-1 for i, s in enumerate(xI) if '1'==s]
    for i in oneL: qc.x(qr[i])
    
    qc.barrier()
    circuit_oracle_2bit(qc,qr,secret)
    qc.barrier()
    qc.measure(qr[2],cr[0])
    return qc

#............................
#............................
def build_condShift_only(xI):
    qr = QuantumRegister(nclbit,'q')
    qc = QuantumCircuit(qr)
    # change input from |0> to |1> as defined by xI
    oneL = [nclbit-i-1 for i, s in enumerate(xI) if '1'==s]
    for i in oneL: qc.x(qr[i])
    
    qc.barrier()
    circuit_condShift_2bit(qc,qr)
    return qc

#............................
#............................
def build_Grover():
    qr = QuantumRegister(nqubit,'q')
    cr = ClassicalRegister(nclbit,'c')
    qc = QuantumCircuit(qr, cr)

    qc.x(qr[nqubit-1]) # set Oracle ancilla (MSB) qbit to 1
    #qc.barrier()
    for j in range(nclbit): qc.h(qr[j])
    qc.h(qr[nqubit-1]) # set Oracle ancilla (MSB) qbit to |->

    # 1 iteration of grover algo
    qc.barrier()
    circuit_oracle_2bit(qc,qr,secret)        
    qc.barrier()
    for j in range(nclbit): qc.h(qr[j])
    circuit_condShift_2bit(qc,qr)
    for j in range(nclbit): qc.h(qr[j])
    qc.barrier()
    # perform measurement on x & y registers, also measure ancila bit=2
    for i in range(nclbit): qc.measure(qr[i], cr[i])
    tot_gates=qc.size()
    print('Grover tot_gates=',tot_gates)

    return qc
    
#...!...!....................
def print_statevect(psiV,text=''):
    psiV=np.asarray(psiV)
    nqubit=psiV.shape[0]//2
    print('  print_statevect,name=%s nqubit=%d'%(text,nqubit))
    reV=np.real(psiV)
    imV=np.imag(psiV)

    ic=0
    for x,y in zip(reV,imV):
        if np.abs(x) <1e-3 :  x=0
        if np.abs(y) <1e-3 : y=0
        ic+=1
        print('   %5.2f %5.2f icol=%d'%(x,y,ic))

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    print('A) - Test oracle w/ secret=',secret)
    for xI in ['00','01','10','11']: # assemble jobs to be submitted for execution
        qc=build_oracle_only(xI)
        print(qc)
        # Compile and run the Quantum circuit on a simulator backend
        job = execute(qc, backendSim,shots=1024)
        countsD=job.result().get_counts(qc)
        print('xI=',xI,' oracle_only counts:',countsD)

    print('\n\nB) - Test conditional phase flip for state=0')
    for xI in ['00','01','10','11']: # assemble jobs to be submitted for execution
        qc=build_condShift_only(xI)
        circF='out/circB%s.pdf'%xI
        print(qc)
        #print('xpdf ',circF)
        job1=execute(qc, backendState)
        psi1V = job1.result().get_statevector()
        print_statevect(psi1V,'inp'+xI)


    print('\n\nC) - run full algo on simulator')
    qc=build_Grover()
    print(qc)

    # Compile and run the Quantum circuit on a simulator backend
    job = execute(qc, backendSim,shots=1024)
    countsD=job.result().get_counts(qc)
    print('full Grover  counts:',countsD)

    circF='out/circGrover_3Qs%s.qasm'%secret
    fd=open(circF,'w')
    fd.write(qc.qasm())
    fd.close()
    print('circ saved to',circF)


'''  EXAMPLE RUN  for secret='01'
    secret= 01
    A) - Test oracle w/ secret= 01
    xI= 00  oracle_only counts: {'0': 1024}
    xI= 01  oracle_only counts: {'1': 1024}
    xI= 10  oracle_only counts: {'0': 1024}
    xI= 11  oracle_only counts: {'0': 1024}

    B) - Test conditional phase flip for input state=0
      print_statevect,name=inp00 nqubit=2
       -1.00  0.00 icol=1
        0.00  0.00 icol=2
        0.00  0.00 icol=3
        0.00  0.00 icol=4
      print_statevect,name=inp01 nqubit=2
        0.00  0.00 icol=1
        1.00  0.00 icol=2
        0.00  0.00 icol=3
        0.00  0.00 icol=4
      print_statevect,name=inp10 nqubit=2
        0.00  0.00 icol=1
        0.00  0.00 icol=2
        1.00  0.00 icol=3
        0.00  0.00 icol=4
      print_statevect,name=inp11 nqubit=2
        0.00  0.00 icol=1
        0.00  0.00 icol=2
        0.00  0.00 icol=3
        1.00  0.00 icol=4

    C) - run full algo on simulator
    full Grover  counts: {'01': 1024}

    D) - run full algo on IBMQX4
    full Grover  counts: {'00': 167, '01': 671, '10': 84, '11': 102}

    Note, state '01' has the most counts - this is the secret to be discovered.
'''
