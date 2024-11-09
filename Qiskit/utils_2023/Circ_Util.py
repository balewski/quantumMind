import numpy as np
from pprint import pprint
import time,  os
import hashlib
import ruamel.yaml  as yaml
from qiskit.converters import circuit_to_dag


#...!...!....................
def do_yield_stats(inpD): # assumes input: {'00':2334,'01':456,... etc}
    # variance of nj-base yield given Ns-shots: var nj = Ns*var <xj> = nj*(Ns-nj)/(Ns-1).
    ''' Steve V: The way I think about this problem is that you have a total of Ns trials.  On any one trial, the probability of the throw ending up in state j is 1 or 0.  The mean probability <xj> of ending up in state j, over Ns tries, is nj/Ns and the uncertainty in that mean probability is given by

var <xj> = Sum(xj-<xj>)^2/[Ns(Ns-1)] = [nj(1-<xj>)^2 + (Ns-nj)<xj>^2]/[Ns(Ns-1)]

This reduces to:

var nj = Ns*var <xj> = nj*(Ns-nj)/(Ns-1).

I think this works for any number of substates and any large number of trials.
    '''
    baseV = sorted(inpD.keys())
    #print(baseV, type(baseV))
    yieldL=[ inpD[x] for  x in baseV ]
    yieldV=np.array(yieldL, dtype=float)
    nShots=sum(yieldV)
    probV=yieldV/nShots
    #print('yieldV, probV',yieldV,probV)

    assert nShots>1
    varL= [  x*(nShots-x)/(nShots-1) for x in yieldL]
    #print('varL',varL)
    #varV=np.array(varL)
    probErrV=np.sqrt(varL)/nShots
    #print('probErrV',probErrV)

    outD={}
    for  x,p,s in zip(baseV,probV,probErrV):
        outD[x]=[inpD[x],p,s]
    #pprint(outD)
    return outD

#...!...!....................
def npAr_avr_and_err(yV, axis=None):
    if axis==None:
        num=yV.size
    else:
        num=yV.shape[axis]
    avr=np.mean(yV,axis=axis)
    std=np.std(yV,axis=axis)
    err=std/np.sqrt(num) # it is error of the avr
    return (avr,std,num,err)
 

#...!...!....................
def read_yaml(ymlFn):
        print('  read  yaml:',ymlFn,end='')
        start = time.time()
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        print(' done, size=%d'%len(bulk),'  elaT=%.1f sec'%(time.time() - start))
        return bulk

#...!...!....................
def write_yaml(rec,ymlFn,verb=1):
        startT = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
            print('  closed:   ',ymlFn,' size=%.2f kB'%xx,'  elaT=%.1f sec'%(time.time() - startT))


#...!...!....................
def save_measFitter_Yaml(meas_fitter,outF,outD={}):
    outD['cal_matrix']=meas_fitter.filter.cal_matrix
    outD['state_labels']=meas_fitter.filter.state_labels
    #print('rrr',str(list(meas_fitter.qubit_list)),str(outD['calibInfo']['qubit_list']))
    assert str(list(meas_fitter.qubit_list))==str(outD['calibInfo']['qubit_list'])
    print('save_measFilter_Yaml keys:',outD.keys())
    write_yaml(outD,outF)

#...!...!....................
def restore_measFitter_Yaml(outF): # overwtites a stub CompleteMeasFitter object
    import copy
    import qiskit
    from qiskit.ignis.mitigation.measurement import complete_meas_cal,  CompleteMeasFitter
    blob=read_yaml(outF)
    qubit_list=blob['calibInfo']['qubit_list']
    nqubit=blob['calibInfo']['nqubit_hw']
    #print('rrrr nqubit=',nqubit, qubit_list)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr= qiskit.QuantumRegister(nqubit), circlabel='jan0cal')
    backend = qiskit.Aer.get_backend('qasm_simulator')
    job3 = qiskit.execute(meas_calibs, backend=backend, shots=5)
    cal_results3 = job3.result()
    # The calibration matrix without noise is the identity matrix
    meas_fitter3 = CompleteMeasFitter(cal_results3, state_labels, qubit_list=qubit_list, circlabel='jan0cal')
    meas_fitter3._tens_fitt.cal_matrices = [copy.deepcopy(blob['cal_matrix'])]

    calibMetaD=blob['calibInfo']
    print('restore_measFitter:',outF); pprint(calibMetaD)
    return meas_fitter3, calibMetaD

#...!...!....................
def load_fake_backend(ibmqN):  # input is real HW name
    import importlib
    assert 'ibmq_' in ibmqN
    core=ibmqN.replace('ibmq_','')
    c=core[0].upper()
    name2='Fake'+c+core[1:]
    #print('qq',core,c,name2)
    #1print('All fake backends:',dir(module))
    module = importlib.import_module("qiskit.test.mock")
    cls = getattr(module, name2)
    return cls()

#...!...!....................
def access_backend(backName='ibmq_santiago', verb=1, fake=False):
    from qiskit import IBMQ
    if 0: # for IBM token go to: https://quantum-computing.ibm.com/account
        #myToken='296..dc6'
        IBMQ.save_account(myToken,overwrite=True)
        print('\nstored account:'); pprint(IBMQ.stored_account())
    
    print('\nIBMQ account - just pass load_account()'); IBMQ.load_account()
    if verb>1:
        print('\nIBMQ providers:'); pprint(IBMQ.providers())
    
    if fake :
        from qiskit.providers.aer import AerSimulator
        if 'ideal' in backName:
            backend= AerSimulator()
        else:            
            device_backend=load_fake_backend(backName)
            backend= AerSimulator.from_backend(device_backend)
            #from qiskit.test.mock import FakeVigo            
            #device_backend = FakeVigo()    
            backend= AerSimulator.from_backend(device_backend)
    else:
        provider = IBMQ.get_provider(group='open')
        if verb>1:
            print('\n  provider beckends'); pprint(provider.backends())
        backend = provider.get_backend(backName)
    print('\nmy backend=',backend)
    backStatus=backend.status().to_dict()
    print(backStatus)
    assert backStatus['operational']
    print('You can proceed on',backName)
    return backend

#...!...!....................
def retrieve_job(backend, job_id, verb=1):
    from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
    print('\nretrieve_job backend=',backend,' search for',job_id)
    job=None
    try:
        job = backend.retrieve_job(job_id)
    except:
        print('job NOT found, quit\n')
        exit(99)
    
    print('job IS found, retrieving it ...')
    startT = time.time()
    job_status = job.status()
    while job_status not in JOB_FINAL_STATES:
        print('Status  %s , est. queue  position %d,  mytime =%d ' %(job_status.name,job.queue_position(),time.time()-startT))
        #print(type(job_status)); pprint(job_status)
        time.sleep(10)
        job_status = job.status()

    print('final job status',job_status)
    if job_status!=JobStatus.DONE :
        print('abnormal job termination', backend,', job_id=',job_id)
        # ``status()`` will simply return``JobStatus.ERROR`` and you can call ``error_message()`` to get more
        print(job.error_message())
        exit(0)
        
    startT = time.time()
    jobHeadD = job.result().to_dict()
    print('job data transfer took %d sec'%(time.time()-startT))
    jobResL=jobHeadD.pop('results')
    print('extracting experiments took %d sec'%(time.time()-startT))
    print('job header:'); pprint(jobHeadD)
    assert jobHeadD['success']

    numExp=len(jobResL)
    print('job completed, num experiments=',numExp,', head_status=',jobHeadD['status'])
    assert numExp>0
    if verb>1:
        iexp=numExp//2
        exp=jobResL[iexp]
        print('example experiment i=',iexp, type(exp)); pprint(exp)

    if len(jobHeadD['job_id'])<1 :
        jobHeadD['job_id']=job.job_id() # Jan:it is missing ??
    return job, jobHeadD, jobResL 


#...!...!....................
def circuit_summary(circ, verb=1):
    dag = circuit_to_dag(circ)
    dagProp=dag.properties()
    #pprint(dagProp)
    ''' Explanation
    -factors = layers=  how many components the circuit can decompose into
    -depth = factors+bariers
    -size = gates+bariers+measuerements
    -qubits = total in the device
    '''
    cnt={'cx':{},'u1':{},'u2':{},'u3':{}}
    outD={}
    for x in ['operations', 'depth','size','factors','qubits']:
        outD[x]=dagProp[x]
    #.... add non-present gates count as 0s
    ciop=outD['operations']
    for x in cnt:
        if x not in ciop: ciop[x]=0
                
    #.... capture used & measure qubits
    gateL=dag.op_nodes()  # will include bariers
    gates=0
    mquL=[]; uquS=set(); mclL=[]

    for idx,node in enumerate(gateL):
        if node.name=='barrier': continue
        if node.name=='measure':
            q=node.qargs[0][1]
            mquL.append(q); uquS.add(q)
            c=node.cargs[0][1]
            mclL.append(c)
            continue
        # below proecess 1 & 2 qubit gates
        gates+=1
        # a) collect all used qubits, to find ancella
        for qu in node.qargs:
             uquS.add(qu.index)

        # b) counnt cx gates for every qubit pair in every direction
        if node.name=='cx':
            qL=[ qu.index for qu in node.qargs]
            qT=tuple(qL)
            if qT not in cnt['cx']: cnt['cx'][qT]=0
            cnt['cx'][qT]+=1
        if node.name in ['u1','u2', 'u3']:
            nm=node.name
            q=node.qargs[0][1]
            if q not in cnt[nm]: cnt[nm][q]=0
            cnt[nm][q]+=1
             
    assert gates>=1
    #print('gate cnt:',cnt)
    outD['gateCnt']=cnt
    #print('used qu=',sorted(list(uquS)))
    outD['meas_clbit']=mclL
    outD['meas_qubit']=mquL
    outD['ancilla']=sorted(list(uquS-set(mquL)))
    outD['gates']=gates
    outD['name']=circ.name
    if verb>0:
        print('\nsummary=',outD)
    #print(outD.keys())
    return outD
    
#...!...!....................
def read_submitInfo(args):
    jid=args.job_id; jid6=args.jid6
    print('verify_jid6 start: job_id=',jid,', jod6=',jid6)
    if jid==None and jid6==None or  jid!=None and jid6!=None:
        print('provide one: jid xor jid6, aborted')
        exit(99)

    # chicken and egg problem
    if jid!=None:
        jid6='j'+jid[-6:]
        print(' will use job_id, proceed')

    if jid6!=None:
        assert jid6[0]=='j'
        print(' will use jid6, proceed')

    inpF=args.dataPath+'submInfo-%s.yaml'%jid6
    blob=read_yaml(inpF)
    args.metaD=blob # contains now: submInfo, expInfo
    print('jobInfo: keys=',blob.keys())

    # fill missing info
    if args.job_id==None:  args.job_id=blob['submInfo']['job_id']
    if args.jid6  ==None:  args.jid6=jid6
    return


#...!...!....................
def circ_2_yaml(circ,verb=1):  # packs circuit data for saving
    outD={}
    outD['info']=circuit_summary(circ,verb)
    outD['qasm']=circ.qasm()

    hao = hashlib.md5(outD['qasm'].encode())
    outD['md5']=hao.hexdigest()
    #pprint(outD)
    return outD
