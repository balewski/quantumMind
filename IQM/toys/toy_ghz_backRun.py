#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister #, transpile
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM,IQMJob
from qiskit.providers.jobstatus import JobStatus
from time import time,sleep
t=time()
sleep(1)

#...!...!....................
def create_ghz(n):    
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)
    
    qc.measure_all()
    return qc

nq=6  ; nshot=2000
qpuName='garnet' ; topo='grid20q'
qpuName='syrius' ; topo='star16q'
#qpuName='deneb' ; topo='star6q'

k0s='0'*nq; k1s='1'*nq

qc=create_ghz(nq)
print(qc)
print('M: ideal circ gates count:', qc.count_ops())

backend1 = AerSimulator()
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))
job = backend1.run(qc, shots=nshot)
result = job.result()
cntD=result.get_counts()
print('Aer:',cntD)

n0=cntD[k0s+' '+k0s]; n1=cntD[k1s+' '+k0s] # hack with extra 0s padding
pure=(n0+n1)/nshot
print('ideal GHZ(%d)  nShot=%d  n0=%d  n1=%d  pure=%.3f'%(nq,nshot,n0,n1,pure))

print('access IQM backend ...',qpuName)
# os.environ["IQM_TOKEN"] is set already
provider=IQMProvider(url="https://cocos.resonance.meetiqm.com/"+qpuName)
backend2 = provider.get_backend()
print('got backend:',backend2.name)

qcT = transpile_to_IQM(qc, backend2)

print(qcT.draw('text', idle_wires=False))
print('M: transpiled GHZ(nq=%d) gates count:'%nq, qcT.count_ops())

job = backend2.run(qcT, shots=nshot)
jid=job.job_id()
print('submitted JID=',jid,backend2.name )

i=0; T0=time()
while True:
    jstat=job.status()
    elaT=time()-T0
    print('P:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
    if jstat in [ JobStatus.DONE, JobStatus.ERROR]: break
    i+=1; sleep(5)
    
print('M: job done, status:',jstat,backend2.name)
  
result = job.result()
cntD=result.get_counts()
if len(cntD) <17: print('cntD:',cntD)
if qpuName=='deneb':
    k0s+='0'*nq
    k1s+='0'*nq

n0=cntD[k0s]; n1=cntD[k1s] # hack with extra 0s padding
pure=(n0+n1)/nshot
print(' %s:%s  GHZ(%d)  nShot=%d  n0=%d  n1=%d  pure=%.3f  numBitStr=%d '%(backend2.name,qpuName,nq,nshot,n0,n1,pure,len(cntD)))

bitStrL=result.get_memory()
print('%d meas bitStr'%len(bitStrL))
print('dump ... %s',bitStrL[:10])

