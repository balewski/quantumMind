#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister #, transpile
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM #,IQMJob
from qiskit.providers.jobstatus import JobStatus
from time import time,sleep
t=time()
sleep(1)


jid='067cb07d-26c0-7544-8000-577afb2edffe' ; qpuName='garnet' # GHZ on Garnet , nq=8?
#jid='067cb071-b79a-7409-8000-98452a7d86c5' ; qpuName='sirius' # GHZ on Sirius


jid='067cb0d9-48a7-7a65-8000-6956dbd8950c'

print('M: access IQM backend ...',qpuName)
provider=IQMProvider(url="https://cocos.resonance.meetiqm.com/"+qpuName)
backend = provider.get_backend()
print('got BCKN:',backend.name,qpuName)

print('M: retrieve jid:',jid)
job=backend.retrieve_job(jid)
jstat=job.status()
print('M: status=%s '%(jstat))


result = job.result()
cntD=result.get_counts()
print('got %d bitStarings'%(len(cntD)))
if len(cntD) <17 or 1: print('cntD:',cntD)
bitStrL=result.get_memory()
print('%d meas bitStr'%len(bitStrL))
print('dump ... %s',bitStrL[:10])

if 0:  exit(0)

# GHZ job only
nq=6  ; nshot=2000
k0s='0'*nq; k1s='1'*nq

if qpuName in [ 'deneb','sirius'] :
    k0s+='0'*nq
    k1s+='0'*nq

n0=cntD[k0s]; n1=cntD[k1s] # hack with extra 0s padding
pure=(n0+n1)/nshot
print(' %s:%s  GHZ(%d)  nShot=%d  n0=%d  n1=%d  pure=%.3f  numBitStr=%d '%(backend.name,qpuName,nq,nshot,n0,n1,pure,len(cntD)))


