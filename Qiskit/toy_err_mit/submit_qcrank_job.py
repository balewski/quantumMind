#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
runs localy  or on HW
tests various error mitigation methods
Records meta-data containing  job_id 
HD5 arrays contain input and output

Use sampler 
Dependence:  qiskit 1.2
Dependency : https://github.com/campsd/data-encoder-circuits
mounted as /qcrank_light

Use case:
  ./run_polyEH_sampler.py    -f exp,2 -k 4   -n 2000  -E


'''
import sys,os,hashlib
import numpy as np
from pprint import pprint
from time import time, localtime

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_Qiskit import  circ_depth_aziz, harvest_circ_transpMeta

from qiskit_aer import AerSimulator
from toolbox.Util_Qiskit import pack_counts_to_numpy
from numpy.polynomial import Polynomial

sys.path.append(os.path.abspath("/qcrank_light"))
from datacircuits import qcrank

import argparse

#...!...!..................
def commandline_parser(backName="aer_ideal",provName="local sim"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--outPath",default='out/',help="all inputs and outputs")

    parser.add_argument("--expName",  default=None,help='(optional) replaces IBMQ jobID assigned during submission by users choice')
 
    # .... QCrank
    parser.add_argument('-q','--numQubits', default=[2,2], type=int,  nargs='+', help='pair: nq_addr nq_data, space separated ')

    parser.add_argument('-i','--numSample', default=21, type=int, help='num of images packed in to the job')

    #parser.add_argument('--numCalibSample', default=0, type=int, help='num of appended  calibration samples, used to improve HW results ')

    # .... job running
    parser.add_argument('-n','--numShot',type=int,default=2000, help="shots per circuit")
    parser.add_argument('-b','--backend',default=backName, help="tasks")
    parser.add_argument( "-B","--noBarrier", action='store_true', default=False, help="remove all bariers from the circuit ")
    parser.add_argument( "-RC","--randomCompile", action='store_true', default=False, help="enable randomized compilation ")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")

    '''there are 3 types of backend
    - run by local Aer:  ideal ,  fake_kyoto
    - submitted to IBM: ibm_kyoto
    '''

    args = parser.parse_args()
    
    args.provider=provName
    if 'ibm' in args.backend:
        args.provider='IBMQ_cloud'
 
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.outPath)
    assert len(args.numQubits)==2
    return args


#...!...!....................
def jan_transpile1(backend,qc):
        
    import mapomatic
    Ta=time()    
    pm1 = generate_preset_pass_manager(optimization_level=3, backend=backend)
    Tb=time()

    # 1st pass, to convert circuit to naitive gates for HW
    qcT1 = pm1.run(qc)
    Tc=time()
    print('transpiled pass1 for',backend,Tb-Ta, Tc-Tb)
    
    #print(qcT1.draw('text', idle_wires=False))
    layout0 = qcT1._layout.final_index_layout(filter_ancillas=True)
    print(' initial phys layout:%s'%(layout0))
    #return qcT1 # plain transpiler
    # remove idle qubits, it does not much for QCrank
    qcT2 = mapomatic.deflate_circuit(qcT1)
   
    # Find all the matching subgraphs of the target backend 
    layouts = mapomatic.matching_layouts(qcT2, backend)
    nLay=len(layouts); mxLay=min(15,nLay)
    print('num layouts:',nLay)
    print('dump few layouts:',layouts[:5])
    #1print(qcT1.draw('text', idle_wires=False))
    #layouts and costs are sorted from lowest to highest. 
    scores = mapomatic.evaluate_layouts(qcT2, layouts, backend)
    for i in range(mxLay):
        rec=scores[i]
        print('rank=%d score=%.3f layout:%s'%(i,rec[1],rec[0]))
    rec=scores[-1]
    print('worst score=%.3f layout:%s'%(rec[1],rec[0]))
        
    # Because the SWAP mappers in Qiskit are stochastic, It is thus beneficial to transpile many instances of a circuit and take the best one.

    layoutBest=scores[0][0]

    if 0:  # KEEP the worst choice
        bqList=backend.properties().general_qlists
        rec=bqList[qc.num_qubits-4]
        print(rec)
        assert rec['name']=='lf_%d'%(qc.num_qubits)
        layoutBest=rec['qubits']
        #[{'name': 'lf_4', 'qubits': [35, 25, 26, 27]}, 

    pm2 = generate_preset_pass_manager(optimization_level=3, backend=backend,initial_layout=layoutBest)
    qcT4 = pm2.run(qcT2)
    return qcT4

    for i in range(mxLay):
        rec=scores[i]
        pm2 = generate_preset_pass_manager(optimization_level=3, backend=backend,initial_layout=scores[i][0])
        qcT4 = pm2.run(qcT2)
        nlg=qcT4.num_nonlocal_gates()
        print('rank=%d score=%.3f nlg=%d layout:%s'%(i,rec[1],nlg,rec[0]))
        
        #physQubitLayout = qcT4._layout.final_index_layout(filter_ancillas=True)
        #print('   phys layout:',physQubitLayout)
        break
    bbb
   
    
    
    

#...!...!....................
def jan_transpile(backend,qc):
    Ta=time()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    Tb=time()
    qcT = pm.run(qc)
    Tc=time()
    print('transpiled2 for',backend,Tb-Ta, Tc-Tb)

   
    return qcT


#...!...!....................
def buildPayloadMeta(args):
    pd={}  # payload
    pd['nq_addr'],pd['nq_data'] = args.numQubits
    pd['num_qubit']=pd['nq_addr']+pd['nq_data']
    pd['num_addr']=2**pd['nq_addr']
    pd['num_pix']=pd['nq_data']*pd['num_addr']

    pd['num_sample']=args.numSample
    pd['max_val']=np.pi   # maximum data value stored at address
    sd={}
    sd['num_shots']=args.numShot

    md={ 'payload':pd, 'submit':sd }
    md['short_name']=args.expName

    if args.verb>1:  print('\nBMD:');pprint(md)
    return md


#...!...!....................
def harvest_sampler_submitMeta(job,md,args):
    sd=md['submit']
    sd['job_id']=job.job_id()
    backN=args.backend
    sd['backend']=backN     #  job.backend().name  V2
    
    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())
    sd['provider']=args.provider
    sd['api_type']='sampler' 
    
    if args.expName==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[3:9] # those are still visible on the IBMQ-web
        tag=args.backend.split('_')[0]
        md['short_name']='%s_%s'%(tag,md['hash'])
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        md['short_name']=args.expName


#...!...!....................
def harvest_sampler_results(job,md,bigD,T0=None):  # many circuits
    qa={}
    jobRes=job.result()
    #print('ddd',jobRes)
    counts=jobRes[0].data.c.get_counts()
    

    if T0!=None:
        elaT=time()-T0
        print(' job done, elaT=%.1f min'%(elaT/60.))
        qa['running_duration']=elaT
    else:
        jobMetr=job.metrics()
        print('jobMetr:',jobMetr)
        qa['timestamp_running']=jobMetr['timestamps']['running']
        qa['quantum_seconds']=jobMetr['usage']['quantum_seconds']
        qa['all_circ_executions']=jobMetr['executions']
        
        if jobMetr['num_circuits']>0:
            qa['one_circ_depth']=jobMetr['circuit_depths'][0]
        else:
            qa['one_circ_depth']=None
    
    pprint(jobRes[0])
    nCirc=len(jobRes)  # number of circuit in the job
    jstat=str(job.status())
    
   
    #print('counts:',counts)
    #print(jobRes[0].data.c0)

    countsL=[ jobRes[i].data.c.get_counts() for i in range(nCirc) ]

    # collect job performance info
    res0cl=jobRes[0].data.c
    qa['status']=jstat
    qa['num_circ']=nCirc
    qa['shots']=res0cl.num_shots
    
    qa['num_clbits']=res0cl.num_bits
    
    print('job QA'); pprint(qa)
    md['job_qa']=qa
    pack_counts_to_numpy(md,bigD,countsL)
    return bigD

#...!...!....................
def ehandsInput_to_qcrankInput(udata):  # shape: n_img,  nq_data, n_addr,
    fdata=np.arccos(udata) # encode user data for qcrank
    # QCrank wants indexs order  to be: n_addr, nq_data, n_img --> (2,1,0)
    fdata=np.transpose(fdata, ( 2,1, 0))
    # QCrank also reverses the order of data qubits
    # Reverse the order of values along axis 1
    fdata = fdata[:, ::-1, :]
    return fdata  # shape : n_addr, nq_data, n_img

#...!...!....................
def bind_random_qcrank_inputs(md,param_qcrank,verb=1):
    pmd=md['payload']
    n_img=pmd['num_sample']
    # generate float random data,  shape: n_img, nq_data, n_addr
    udata = np.random.uniform(-1,1, size=( n_img,  pmd['nq_data'], pmd['num_addr']))
    fdata=ehandsInput_to_qcrankInput(udata)  # shape : n_addr, nq_data, n_img
    if args.verb>2:
        print('input data=',data.shape,repr(udata))
    
    # bind the data
    param_qcrank.bind_data(fdata, max_val= pmd['max_val'])
    # generate the instantiated circuits
    qcL = param_qcrank.instantiate_circuits()
  
    expD={'inp_udata': udata, 'fdata':fdata}
    return qcL,expD

#...!...!....................
def make_qcrank_obj(md,verb=1):
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    pmd=md['payload']
    # set up experiments
    qcrank_obj = qcrank.ParametrizedQCRANK(
        pmd['nq_addr'],
        pmd['nq_data'],
        qcrank.QKAtan2DecoderQCRANK,
        keep_last_cx=True, # keep the last cnot
        measure=True,
        statevec=False,
        reverse_bits=True,   # to match Qiskit littleEndian
        parallel= True     # T: optimal,  F: not optimal w/ cx-gates being parallele
    )
    # rename classical register for consistency with transpiled circuit
    qr = QuantumRegister(pmd['num_qubit'], 'q')
    cr = ClassicalRegister(pmd['num_qubit'], 'c')
    qc= QuantumCircuit(qr, cr)
    qcrank_obj.circuit=qc.compose(qcrank_obj.circuit)#, inplace=True)
    
    return qcrank_obj

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    MD=buildPayloadMeta(args)
   
    pprint(MD)
   
    qcrank_obj=make_qcrank_obj(MD)
    qcP=qcrank_obj.circuit
    cxDepth=qcP.depth(filter_function=lambda x: x.operation.name == 'cx')
 
    nqTot=qcP.num_qubits
        
    print('M: circuit has %d qubits, %d CX-depth'%(nqTot,cxDepth), qcP.name)
    depthPC,opsPC=circ_depth_aziz(qcP,text='circ_orig')

    
    if args.verb>0 and qcP.num_qubits<=5 or args.verb>1 :
        print('M:.... PARAMETRIZED IDEAL CIRCUIT ..............')
        print(qcP)
        
        #print(qcP.draw('text', idle_wires=False)) #,cregbundle=True

    runLocal=True

    # ------  construct sampler(.) job ------
    if 'ideal' in args.backend: 
        qcT=qcP
        transBackN='ideal'
        backend = AerSimulator()
    else:
        print('M: activate QiskitRuntimeService() ...')
        #Ta=time()
        service = QiskitRuntimeService()
        #Tb=time(); print('M:act t:',Tb-Ta)
        if  'fake' in args.backend:
            transBackN=args.backend.replace('fake_','ibm_')
            Ta=time()
            noisy_backend = service.backend(transBackN)
            Tb=time();
            backend = AerSimulator.from_backend(noisy_backend)
            Tc=time(); print('M:aer(fake) t:',Tb-Ta,Tc-Tb)
            print('fake noisy backend =', noisy_backend.name)
            qcT = jan_transpile(backend,qcP)
        else:
            assert 'ibm' in args.backend
            backend = service.backend(args.backend)  # overwrite Aer-backend
            print('use HW backend =', backend.name)
            transBackN=backend.name
            transp_backend =backend
            runLocal=False
            qcT = jan_transpile(backend,qcP)
        #1print(qcT.draw('text', idle_wires=False))
        #1 print(qcT)

    if args.verb>0 and qcP.num_qubits<=5 or args.verb>1 :
        print('M:.... PARAMETRIZED Transpiled CIRCUIT ..............')
        #1print(qcT.draw('text', idle_wires=False))
        

    circ_depth_aziz(qcP,'ideal')
    circ_depth_aziz(qcT,'transpiled')
    harvest_circ_transpMeta(qcT,MD,transBackN)

    # -------- bind the data to parametrized circuit  -------
    qcrank_obj.circuit=qcT
    qcEL,expD=bind_random_qcrank_inputs(MD,qcrank_obj)
  
   
    print('M: run on backend:',backend.name)
       

    nCirc=args.numSample
   
    if args.verb>1: print(qcEL[0].draw('text', idle_wires=False))
    print('M: execution-ready %d circuits with %d qubits backend=%s'%(nCirc,nqTot,backend.name))
                            
    if not args.executeCircuit:
        pprint(MD)
        print('\nNO execution of circuit, use -E to execute the job\n')
        exit(0)

    # ----- submission ----------
    numShots=MD['submit']['num_shots']
    print('M:job starting, nCirc=%d  nq=%d  shots/circ=%d at %s  ...'%(nCirc,qcEL[0].num_qubits,numShots,args.backend),backend)


    options = SamplerOptions()
    options.default_shots=numShots
    if 0:
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True
        options.twirling.num_randomizations=20
        #options.twirling.startegy='all'
    
    print('opp', options.twirling)        
        
    sampler = Sampler(mode=backend, options=options)
    T0=time()
    #1print(qcEL[0].draw('text', idle_wires=False))
    
    job = sampler.run(tuple(qcEL), shots=numShots)
    Td=time(); print('M: td:',Td-T0)
    harvest_sampler_submitMeta(job,MD,args)    
    if args.verb>1: pprint(MD)

    
    if runLocal:
        harvest_sampler_results(job,MD,expD,T0=T0)
        print('M: got results')
        #...... WRITE  MEAS OUTPUT .........
        outF=os.path.join(args.outPath,MD['short_name']+'.meas.h5')
        write4_data_hdf5(expD,outF,MD)        
        print('   ./postproc_qcrank.py  --expName   %s   -p a    -Y\n'%(MD['short_name']))
    else:
        #...... WRITE  SUBM OUTPUT .........
        outF=os.path.join(args.outPath,MD['short_name']+'.ibm.h5')
        write4_data_hdf5(expD,outF,MD)
        #print('M:end --expName   %s   %s  %s  jid=%s'%(expMD['short_name'],expMD['hash'],backend.name ,expMD['submit']['job_id']))
        print('   ./retrieve_ibmq_sampler.py --expName   %s   \n'%(MD['short_name'] ))




   
