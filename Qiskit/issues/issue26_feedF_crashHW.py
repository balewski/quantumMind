#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
construct array of feed-forward circuits

'''
import numpy as np
from pprint import pprint

from qiskit.tools.visualization import circuit_drawer
from qiskit import  transpile
from qiskit_ibm_runtime import QiskitRuntimeService

from qiskit import QuantumCircuit,ClassicalRegister, QuantumRegister
import sys,os
from time import time
from qiskit.circuit import Parameter,ParameterVector


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-i','--numSample', default=20, type=int, help='num of images packed in to the job')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=5000, help="shots per circuit")
    parser.add_argument('-b','--backend',default='ibm_torino',   help="backend for transpiler  or 'aer' " )
   
    args = parser.parse_args()
    # make arguments  more flexible
    args.numInput=3
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    return args



#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['num_input']=args.numInput
    pd['num_sample']=args.numSample
    pd['activation']='one'   
    md={ 'payload':pd}
    md['short_name']='test3inp'

    if args.verb>1:  print('\nBMD:');pprint(md)            
    return md

#...!...!....................
def add_HadamardTest(qc,qs,qh,cr):
    qc.h(qh)
    qc.cx(qs,qh)   
    qc.sdg(qh)
    qc.h(qh)
    qc.measure(qh, cr)
    with qc.if_test((cr,1)):  # good for scaling
        qc.z(qs)
    qc.reset(qh)
  

#...!...!....................
def addEhandUnit(qc,mathOp,qrXa,qrXb,gain, unitId, qrZa=None, crZa=None, qrZb=None, crZb=None,**kwargs):
    print('AEU:: %s uid=%d inp: qrXa=%s   qrXb=%s  gain=%.3f'%(mathOp,unitId,qrXa,qrXb,gain))
    assert mathOp in ['sum','prod']

    # add *stabilizers* for the inputs
    if qrZa!=None: # collapse imaginary component of input A
        add_HadamardTest(qc,qrXa,qrZa,crZa)
        qc.barrier(qrXa,qrZa)
    if qrZb!=None: # collapse imaginary component of input B
        add_HadamardTest(qc,qrXb,qrZb,crZb)
        qc.barrier(qrXb,qrZb)
    
    # ... ctrl(0)-Rz(pi,1) ...
    qc.rz(np.pi/2,qrXb)
    qc.cx(qrXa,qrXb)
    qc.rz(-np.pi/2,qrXb)
    
    # ... ctrl(1)-Ry(alpha,0)
    alpha=np.arccos( 1-2*gain)
    qc.ry(alpha/2,qrXa)
    qc.cx(qrXb,qrXa)
    qc.ry(-alpha/2,qrXa)

    if mathOp =='sum':
        qc.reset(qrXb)
        return qrXa,qrXb
    else:
        qc.reset(qrXa)
        return qrXb,qrXa
    


#...!...!....................
def circ_QNeuron(md): 
    pmd=md['payload']
    n_inp=pmd['num_input']
    n_img=pmd['num_sample']

    GHang=np.pi/2 # np.arccos(1.- 2* 0.5); g=1/2 

    nqreg_data=2*n_inp
    XangP=ParameterVector('Xang',length=n_inp)
    WangP=ParameterVector('Wang',length=n_inp)
    BangP=Parameter('Bang')
    
    qrd=QuantumRegister(nqreg_data+1) # data qubits & bias
    cra=ClassicalRegister(3)  # auxiliary cbits
    cro=ClassicalRegister(1)  # output qubits
    
    qc = QuantumCircuit(qrd,cro,cra)
    uid=0

    #.... placholders for 2nd layer of blocks
    qrOut=[]; qrAux=[]

    #.... input layer .....
    
    opts={'gain':0.5, 'mathOp':'prod' }
    for k in range(n_inp):
        j=2*k
        opts['unitId']=k
        opts['qrXa']=qrd[j]
        opts['qrXb']=qrd[j+1]

        qc.ry(XangP[k],opts['qrXa'])
        qc.ry(WangP[k],opts['qrXb'])

        qrO,qrA=addEhandUnit(qc,**opts)
        qrOut.append(qrO)
        qrAux.append(qrA)

    qrx=qrd[nqreg_data]
    qc.ry(BangP,qrx) # encode bias
    qrOut.append(qrx)    
    nOut=len(qrOut)
    qc.barrier()

    assert nOut==4
    # 2nd layer of sumation    w bias      
    assert len(qrAux)>=2
    opts={'gain':0.5, 'mathOp':'sum' }
    #... merge inputs 0+1
    opts['unitId']=k+1
    opts['qrXa']=qrOut[0]; opts['qrZa']=qrAux[0]; opts['crZa']=cra[0]
    opts['qrXb']=qrOut[1]; opts['qrZb']=qrAux[1]; opts['crZb']=cra[1]        
    qrO,qrA=addEhandUnit(qc,**opts)
    qrOut2=[qrO]  # tmp
  
    #... merge inputs 2+bias
    opts['unitId']=k+2
    opts['qrXa']=qrOut[2]; opts['qrZa']=qrAux[2]; opts['crZa']=cra[2]
    opts['qrXb']=qrOut[3]; opts['qrZb']=None
    qrO,qrA=addEhandUnit(qc,**opts)
    qrOut2.append(qrO)
    qrOut=qrOut2  # overwrite output register
    nOut=len(qrOut)
    qc.barrier()

    print('nnn',nOut)
    assert  nOut==2
    # 2nd layer of sumation    w/o bias      
    assert len(qrAux)>=2
    opts={'gain':0.5, 'mathOp':'sum' }
    opts['unitId']=k+1
    opts['qrXa']=qrOut[0]; opts['qrZa']=qrAux[0]; opts['crZa']=cra[0]
    opts['qrXb']=qrOut[1]; opts['qrZb']=qrAux[1]; opts['crZb']=cra[1]

    qrO,qrA=addEhandUnit(qc,**opts)
    qrOut=[qrO]  # tmp
    qrAux=[qrA]
    qc.barrier()
       
    #.... final measurement
    
    assert len(qrOut)==1    
    qc.measure(qrOut[0],cro)
   
    parD={'Xang':XangP, 'Wang':WangP,'Bang':BangP}
    return qc,parD
   

#...!...!....................
def add_HadamardTestB(qc,qs,qh,cr):
    qc.h(qh)
    qc.cx(qs,qh)   
    #qc.sdg(qh)
    qc.h(qh)
    qc.measure(qh, cr)
    return
    with qc.if_test((cr,1)):  # good for scaling
        #qc.z (qs)
        qc.x (qs)
        #qc.measure(qh, crr)
        #qc.rz(np.pi/2,qs)
    qc.reset(qh)
  

#...!...!....................
def construct_random_input(md):
    pmd=md['payload']
    n_inp=pmd['num_input']
    n_img=pmd['num_sample']
    pprint(md)
    
    # generate user data , float random
    eps=1e-4 # just in case
    Xdata = np.random.uniform(-1.+eps, 1.-eps, size=(n_img,n_inp ))  # inputs
    Wdata = np.random.uniform(-1.+eps, 1.-eps, size=(n_img,n_inp))  # weights
    Bdata = np.random.uniform(-1.+eps, 1.-eps, size=(n_img))        # biases

    Xang=np.arccos(Xdata) # encode  data
    Wang=np.arccos(Wdata) 
    Bang=np.arccos(Bdata) 
 
    print('input Xdata=',Xdata.shape,'\n',repr(Xdata[...,:3].T))
    outD={'Xdata':Xdata,'Xang':Xang, 'Wdata':Wdata,'Wang':Wang,   'Bdata':Bdata,'Bang':Bang}

    # Element-wise multiplication
    prodV = np.multiply(Xdata, Wdata)
    
    # Sum along the first axis (rows)
    XxW = np.sum(prodV, axis=1)+Bdata
    #print('XxW',XxW.shape,XxW)
    Tdata=XxW/ (n_inp+1) # because inp values + bias  were averaged

    print('Tdata:', Tdata.shape,prodV.shape)
    outD['Tdata']=Tdata
    
    # Check if the array contains any NaN values
    has_nan = np.isnan(Xang).any()
    if has_nan:
        print('NaN detected in CICT1, dump and abort')
        pprint(outD)
        print('NaN detected in CICT1, ABORT')
        exit(99)
    
    return outD


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    MD=buildPayloadMeta(args)
    #pprint(MD)
   
    expD=construct_random_input(MD)
    
    #....  circuit generation .....
    qcP,paramD=circ_QNeuron(MD)
    print(qcP.draw(output='text',idle_wires=False))  # skip ancilla

    print('M: acquire backend:',args.backend)
    if args.backend=='aer':
         from qiskit import   Aer
         backend = Aer.get_backend('aer_simulator')
    else:
        print('connecting to IBM...')
        service = QiskitRuntimeService()
        backend = service.get_backend(args.backend)
        
    if "if_else" not in backend.target:
        from qiskit.circuit import IfElseOp
        backend.target.add_instruction(IfElseOp, name="if_else")

    
    qcT = transpile(qcP, backend=backend, optimization_level=3, seed_transpiler=44)
    print(qcT.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla
 
    #... collect needed params and clone circuits
    qcEL=[ None for i in range(args.numSample) ]
    for i in range(args.numSample):
        cparamD={ paramD[xx]:expD[xx][i] for xx in paramD}
        if i<5: print('M:mapped params',i, cparamD,'\n')
        # Bind the values to the circuit
        qc1=qcT.assign_parameters(cparamD)
        qcEL[i]=qc1
        

    print(qc1.draw(output='text',idle_wires=False,cregbundle=False))  # skip ancilla
    nCirc=len(qcEL)
   
    print('job started, nCirc=%d  nq=%d  shots/circ=%d at %s ...'%(nCirc,qcEL[0].num_qubits,args.numShot,backend))
    T0=time()
    job = backend.run(qcEL,shots=args.numShot) 
    result=job.result()
    elaT=time()-T0
    print('M:  ended elaT=%.1f sec'%(elaT))
    probsBL=[{'0':0,'1':0}  for ic in range(nCirc)]
    for ic in range(nCirc):
        counts = result.get_counts(ic)        
        print('dump ic=',ic); pprint(counts)
  
    print('M:done')
