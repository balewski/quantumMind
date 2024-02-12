#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Implements n-bit Grover algo 
 *) arbitrary n-bit secret
 *) n-bit oracle is build as a nested Toffoli gates 
 *) C^nX gate is build  using  Nielsen & Chuang construct
'''

import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import Aer, execute

backendSim=Aer.get_backend("qasm_simulator")
backendState=Aer.get_backend("statevector_simulator")


# CHANGE ONLY NEXT LINE
#nclbit=2 ; secret='01' ; iterG=1  # count(secret=01)=8192,  prob=1.000
nclbit=3 ; secret='011' ; iterG=2  # count(secret=011)=7731,  prob=0.944  @2 iter, 1.0 @6 iter
#nclbit=4 ; secret='1001' ; iterG=3  # count(secret=1011)=7895,  prob=0.964 @3 iter, 0.992 @ 9 iter 
#nclbit=5 ; secret='10110' ; iterG=4 # count(secret=10110)=8187,  prob=0.999 @4 iter
#nclbit=8 ; secret='00011010' ; iterG=5 #  iG=4:p=0.29, iG=5:p=0.40, iG=6:p=0.52, iG=7:p=0.65
#nclbit=10 ; secret='0000110101' ; iterG=5 # if larger :"CIRCUIT_SIZE_EXCEEDED"


# do NOT change code below (too much)
nShots=1024*8
assert len(secret)==nclbit
nqubit=2*nclbit-1
print('Exercise %d-bit  Grover search for secret=%s'%(nclbit,secret))


#............................
#............................
#
def circuit_oracle_nclbit(qc,qr,secret):
    nanc=nclbit-2 # ancilla qubits
    # the MSB qubit will be the output of oracle
    #print('Or: secret',secret,' inp=',nclbit,' anc=',nanc)

    zerL = [nclbit-i-1 for i, s in enumerate(secret) if '0'==s]
    
    for i in zerL:  qc.x(qr[i])  # flip 0-s to 1
    #qc.barrier()

    # build C^n using  Nielsen & Chuang construct
    assert nclbit>=2 # 1-bit secret is too simple
    if nanc==0: # nclbit==2
        qc.ccx(qr[0], qr[1], qr[2])
    else:
        qc.ccx(qr[0], qr[1], qr[nclbit])
        for j in range(0,nanc):
            qc.ccx(qr[2+j], qr[nclbit+j], qr[nclbit+j+1])
        # undo ancilla
        for j in range(nanc-1,0,-1):
            qc.ccx(qr[1+j], qr[nclbit+j-1], qr[nclbit+j])
        qc.ccx(qr[0], qr[1], qr[nclbit])
    #qc.barrier()
    for i in zerL:  qc.x(qr[i])  # undo flip of 0-s


#............................
#............................
def circuit_condShift_nclbit(qc,qr):
    nanc=nclbit-3 # ancilla qubits
    #print('CNSh: secret inp=',nclbit,' anc=',nanc)
    # the MSB qubit will be the output of oracle
    assert nclbit>=2 # 1-bit secret not implemented

    for i in range(nclbit): qc.x(qr[i])
    qc.h(qr[nclbit-1]) # H for MSB of input

    if nclbit==2:
        qc.cx(qr[0],qr[1])
    elif nclbit==3:
         qc.ccx(qr[0],qr[1],qr[2])
    else:
        qc.ccx(qr[0], qr[1], qr[nclbit+nanc-1])
        for j in range(0,nanc):
            qc.ccx(qr[2+j], qr[nclbit+nanc-1-j], qr[nclbit+nanc-2-j])
        # undo ancilla
        for j in range(nanc-1,0,-1):
            qc.ccx(qr[1+j], qr[nclbit+nanc-j], qr[nclbit+nanc-1-j])
        qc.ccx(qr[0], qr[1], qr[nclbit+nanc-1])
    #qc.barrier()
    qc.h(qr[nclbit-1]) # H for MSB of input
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
    circuit_oracle_nclbit(qc,qr,secret)
    qc.barrier()
    qc.measure(qr[nqubit-1],cr[0])
    return qc

#............................
#............................
def build_condShift_only(xI):
    if nclbit<=3:
        nqubit=nclbit
    else:
        nqubit=2*nclbit-3

    qr = QuantumRegister(nqubit,'q')
    qc = QuantumCircuit(qr)
    # change input from |0> to |1> as defined by xI
    oneL = [nclbit-i-1 for i, s in enumerate(xI) if '1'==s]
    for i in oneL: qc.x(qr[i])
    
    qc.barrier()
    circuit_condShift_nclbit(qc,qr)
    return qc

#............................
#............................
def build_Grover(iterG):
    qr = QuantumRegister(nqubit,'q')
    cr = ClassicalRegister(nclbit,'c')
    qc = QuantumCircuit(qr, cr)

    qc.x(qr[nqubit-1]) # set Oracle ancilla (MSB) qbit to 1
    #qc.barrier()
    for j in range(nclbit): qc.h(qr[j])
    qc.h(qr[nqubit-1]) # set Oracle ancilla (MSB) qbit to |->

    for itg in range(iterG):    # 1 iteration of grover algo
        qc.barrier()
        circuit_oracle_nclbit(qc,qr,secret)        
        qc.barrier()
        for j in range(nclbit): qc.h(qr[j])
        circuit_condShift_nclbit(qc,qr)
        for j in range(nclbit): qc.h(qr[j])
    qc.barrier()
    # perform measurement on x & y registers
    for i in range(nclbit): qc.measure(qr[i], cr[i])
    tot_gates=qc.size()
    print('Grover size=',tot_gates)
    return qc



#...!...!....................
def print_statevect(psiV,text='',verb=0):
    nqubit=psiV.shape[0]//2
    print('  print_statevect,name=%s nqubit=%d verb=%d'%(text,nqubit,verb))
    reV=np.real(psiV)
    imV=np.imag(psiV)

    ic=0
    for x,y in zip(reV,imV):
        nZero=0
        if np.abs(x) <1e-3 :
                x=0; nZero+=1
        if np.abs(y) <1e-3 :
                y=0; nZero+=1
        ic+=1
        if verb>0 or nZero<2:
            print('   %5.2f %5.2f icol=%d'%(x,y,ic))

#...!...!....................
def get_compiledQasm(qc,backend,backendConfig,seed):
    coupling_map = backendConfig['coupling_map']
    circ_compiled = compile(qc, backend=backend, coupling_map=coupling_map, seed=seed)

    modelQasm=qc.qasm().split('\n')
    rawTxt=circ_compiled.experiments[0].header.compiled_circuit_qasm
    compiQasm=rawTxt.split('\n')
    return  modelQasm,compiQasm

#...!...!....................
def circuit_summary(name,qasmL, verb=0):
    cnt={'qbit':0, 'gate':0, 'cbit':0,'barr':0,'meas':0}
    for rec in qasmL:
        if 'qreg' in rec : cnt['qbit']=int(rec[7:-2])
        if 'creg' in rec : cnt['cbit']=int(rec[7:-2])
        if 'h'==rec[:1] : cnt['gate']+=1
        if 'x'==rec[:1] : cnt['gate']+=1
        if 'u'==rec[:1] : cnt['gate']+=1
        if 'cx'==rec[:2] : cnt['gate']+=1
        if 'id'==rec[:2] : cnt['gate']+=1
        if 'ccx'==rec[:3] : cnt['gate']+=1
        if 'barr'==rec[:4] : cnt['barr']+=1
        if 'meas'==rec[:4] : cnt['meas']+=1
        #print (rec)
    if verb:
        print('circ=',name,' summary:',cnt)
    return cnt

#=================================
#=================================
#  M A I N 
#=================================
#=================================

print('nA) - Test oracle w/ secret=',secret)

for iI in range(2**nclbit): # assemble jobs to be submitted for execution
    #break
    xI=("{0:{fill}%db}"%(nclbit)).format(iI, fill='0')
    qc=build_oracle_only(xI)
    if iI<10: print(qc)
    # Compile and run the Quantum circuit on a simulator backend
    job = execute(qc, backendSim,shots=nShots)
    countsD=job.result().get_counts(qc)
    print('xI=',xI,' oracle_only counts:',countsD,'\n - - - - - - ')

    
print('\nB) - Test conditional phase flip for state=0')
for iI in range(2**nclbit): # assemble jobs to be submitted for execution
    #break
    xI=("{0:{fill}%db}"%(nclbit)).format(iI, fill='0')
    qc=build_condShift_only(xI)
    if iI<10: print(qc)
    job1=execute(qc, backendState)
    psi1V = job1.result().get_statevector()
    print_statevect(psi1V,'inp'+xI)
    #okok


print('\nSimulate  full Grover %dQ-secret algo, num grover iter=%d'%(nclbit,iterG))
qc=build_Grover(iterG)
print(qc)

# Compile and run the Quantum circuit on a simulator backend
job = execute(qc, backendSim,shots=nShots)
countsD=job.result().get_counts(qc)
solution=countsD[secret]
print('full Grover %d iterations,  count(secret=%s)=%d,  prob=%.3f, shots=%d'%(iterG,secret,solution, solution/nShots, nShots))
print('all counts:',countsD)

circF='out/circGrover_%dQs%si%d.qasm'%(nqubit,secret,iterG)
fd=open(circF,'w')
fd.write(qc.qasm())
fd.close()
print('circ saved to',circF)




                                
    


# - - - - - - - - - ---- 

'''  EXAMPLE RUN  for secret='011'
Exercise 3-bit Grover search for secret= 011
A) - Test oracle w/ secret= 011
xI= 000  oracle_only counts: {'0': 1024}
xI= 001  oracle_only counts: {'0': 1024}
xI= 010  oracle_only counts: {'0': 1024}
xI= 011  oracle_only counts: {'1': 1024}
xI= 100  oracle_only counts: {'0': 1024}
xI= 101  oracle_only counts: {'0': 1024}
xI= 110  oracle_only counts: {'0': 1024}
xI= 111  oracle_only counts: {'0': 1024}
B) - Test conditional phase flip for state=0
  print_statevect,name=inp000 nqubit=4 verb=0
   -1.00  0.00 icol=1
  print_statevect,name=inp001 nqubit=4 verb=0
    1.00  0.00 icol=2
  print_statevect,name=inp010 nqubit=4 verb=0
    1.00  0.00 icol=3
  print_statevect,name=inp011 nqubit=4 verb=0
    1.00  0.00 icol=4
  print_statevect,name=inp100 nqubit=4 verb=0
    1.00  0.00 icol=5
  print_statevect,name=inp101 nqubit=4 verb=0
    1.00  0.00 icol=6
  print_statevect,name=inp110 nqubit=4 verb=0
    1.00  0.00 icol=7
  print_statevect,name=inp111 nqubit=4 verb=0
    1.00  0.00 icol=8
C) - run full Grover algo on simulator, num iterations= 1
Grover tot_gates= 25
full Grover count(secret=011)=803,  prob=0.784
full Grover all  counts: {'011': 803, '110': 26, '010': 37, '000': 33, '101': 29, '111': 29, '100': 35, '001': 32}
'''
