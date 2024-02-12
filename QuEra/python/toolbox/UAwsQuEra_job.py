__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from pprint import pprint
import time
import os, hashlib
from braket.aws.aws_quantum_task import AwsQuantumTask
import json # tmp for raw data storing
import numpy as np
import pytz  # for time zones
from datetime import datetime
from collections import Counter
from toolbox.Util_miscIO import dateT2Str
from bitstring import BitArray

#...!...!..................
def submit_args_parser(backName,parser=None,verb=1):
    if parser==None:  parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")
    parser.add_argument("--expName",  default=None,help='(optional) replaces IBMQ jobID assigned during submission by users choice')
        
    # Braket:
    parser.add_argument('-b','--backendName',default=backName,choices=['cpu','emulator','qpu','aquila','sv1','amazon_sv1'],help="backend for computations " )

    parser.add_argument('-n','--numShots', default=23, type=int, help='num of shots')
    parser.add_argument('--evol_time_us', default=1.2, type=float, help='Hamiltonian evolution time, in usec')
    parser.add_argument('--rabi_ramp_time_us', default=[0.10], type=float, nargs='+', help='Rabi ramp up, down time, in usec, space separated list')
    parser.add_argument('--rabi_omega_MHz', default=2.5, type=float, help='Rabi drive frequency in MHz')
    parser.add_argument('--detune_delta_MHz', default=[-6.3,6.3],  nargs='+', type=float, help='detune min-max Rabi  frequency in MHz,  space separated list') 
    parser.add_argument( "-E","--executeTask", action='store_true', default=False, help="may take long time, test before use ")
     
    parser.add_argument( "-C","--cancelTask", action='store_true', default=False, help="instant task cancelation, usefull for debugging of the code")
    parser.add_argument( "-F","--fakeSubmit", action='store_true', default=False,help="do not submit the task to AWS but create job.h5 file")
        
    args = parser.parse_args()
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.outPath=os.path.join(args.basePath,'jobs')
        
    if args.backendName=='cpu':  args.backendName='emulator'
    if args.backendName=='qpu':  args.backendName='aquila'
    if args.backendName=='sv1':  args.backendName='amazon_sv1'
    if len(args.rabi_ramp_time_us)==1:
        tt=args.rabi_ramp_time_us[0]
        args.rabi_ramp_time_us=[tt,tt]
    if len(args.detune_delta_MHz)==1:
        tt=args.detune_delta_MHz[0]
        args.detune_delta_MHz=[-tt,tt]
    assert not (args.executeTask and args.fakeSubmit)
    
    for arg in vars(args):
        if verb==0: break
        print( 'myArgs:',arg, getattr(args, arg))
    
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def access_quera_device( backName,verb=1):
    if backName=='emulator':
         from braket.devices import LocalSimulator
         device = LocalSimulator("braket_ahs")
    if backName=='aquila':
        from braket.aws import AwsDevice
        device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
        
    if backName=='amazon_sv1':
        from braket.aws import AwsDevice
        device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        
    return device
   

#...!...!.................... 
def harvest_submitInfo(job,md,taskName='exp'):
    sd=md['submit']
  
    if job!=None:   # real HW
        sd['task_arn']=job.id
        awsMD=job.metadata()
        for x in [ 'deviceArn', 'status']:
            sd[x]=awsMD[x]
        x='createdAt'
        sd[x]=str(awsMD[x])
    else:  # emulator
        sd['task_arn']='quantum-task/emulator-'+hashlib.md5(os.urandom(32)).hexdigest()[:6]
        sd['deviceArn']='local-emulator'    
     
    t1=time.localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time.time())
    if md['short_name']==None :
        # the  6 chars in job id is a handy job identiffier
        md['hash']=sd['task_arn'].replace('-','')[-6:] # those are still visible on the IBMQ-web
        name=taskName+'_'+md['hash']
        md['short_name']=name
    else:  # name provided or emulator
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
    sd['info']='job:%s,back:%s'%(md['short_name'],md['submit']['backend'])

                
#...!...!.................... 
def harvest_retrievInfo(jobMD,md):
    # collect job performance info    
    qaD={}
    qaD['success']=True
    if jobMD==None:  # emulator
        qaD['status']='COMPLETED'
        t1=time.localtime()
        qaD['exec_date']=dateT2Str(t1)
    else:
        qaD['status']=jobMD['status']
        qaD['endedAt']=str(jobMD['endedAt'])
        assert md['submit']['task_arn'] == jobMD['quantumTaskArn'], 'messed up job ID'
        #1print('aa1',md['submit']['task_arn'])
        #1print('aa2',jobMD['quantumTaskArn']) #; pprint(jobMD)
        
        xS=qaD['endedAt']
        tOb = datetime.fromisoformat(xS)
        #print('tOb',tOb)
        #tS=tOb.strftime("%Y%m%d_%H%M%S_%Z")
        #print('tS UTC',tS)
    
        # Convert the datetime object to the Pacific Daylight Time (PDT) timezone
        pdt_timezone = pytz.timezone('US/Pacific')
        tOb2 = tOb.astimezone(pdt_timezone)
        tS2=tOb2.strftime("%Y%m%d_%H%M%S_%Z")
        
        qaD['exec_date']=tS2
    qaD['info']='exec,%s'%(qaD['exec_date'])
    print('job QA',qaD, 'backend=',md['submit']['backend'])
    md['job_qa']=qaD



#...!...!....................
def retrieve_aws_job( task_arn, verb=1):  # AWS-QuEra job retrieval
    #print('\nretrieve_job search for ARN',task_arn)

    job=None
    try:
        job = AwsQuantumTask(arn=task_arn, poll_timeout_seconds=30)
    except:
        print('job NOT found, quit\n ARN=',task_arn)
        exit(99)

    print('job IS found, state=%s, %s  retriving ...'%(job.state(),task_arn))    
    startT = time.time()
    
    while job.state()!='COMPLETED':        
        if job.state() in ['CANCELLING', 'CANCELLED', 'FAILED']:
            print('retrieval ABORTED for ARN:%s\n'%task_arn)
            exit(0)

        print('State  %s,    mytime =%d, wait ... ' %(job.state(),time.time()-startT))
        time.sleep(30)
        
    print('final job status:',job.state())
        
    return  job 


"""Aggregate state counts from AHS shot results.

        Returns:
            Dict[str, int]: number of times each state configuration is measured.
            Returns None if none of shot measurements are successful.
            Only succesful shots contribute to the state count.

        Notes: We use the following convention to denote the state of an atom (site):
            e: empty site
            r: Rydberg state atom
            g: ground state atom
"""

#...!...!..................
def postprocess_job_results(rawCounts,md,expD,rawShots=None):
    # for now just save Dict[str, int] as JSON    
    expD['counts_raw.JSON']=json.dumps(rawCounts)  #  dictionary of egr-strings counts
    if rawShots!=None: expD['shots_raw.JSON']=json.dumps(rawShots) #   list of egr-strings

    nSol=len(rawCounts)
    md['job_qa']['num_sol']=nSol
    print("post proc., numSol=%d"%(nSol))

    
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import  AnalogHamiltonianSimulationShotStatus

#...!...!..................
def hacked_get_shots(measurements):
    ''' see my ticket https://github.com/amazon-braket/amazon-braket-examples/issues/369
    This code has been hacked from get_counts()
https://amazon-braket-sdk-python.readthedocs.io/en/latest/_modules/braket/tasks/analog_hamiltonian_simulation_quantum_task_result.html#AnalogHamiltonianSimulationQuantumTaskResult.get_counts
 
        Returns:
            list of bistrings per shot
    '''


    state_list=[]
    states = ["e", "r", "g"]
    for shot in measurements:
        if shot.status == AnalogHamiltonianSimulationShotStatus.SUCCESS:
            pre = shot.pre_sequence
            post = shot.post_sequence
            # converting presequence and postsequence measurements to state_idx
            state_idx = [
                0 if pre_i == 0 else 1 if post_i == 0 else 2 for pre_i, post_i in zip(pre, post)
            ]
            state = "".join(map(lambda s_idx: states[s_idx], state_idx))
            state_list.append(state)

    return state_list


#...!...!..................
def sort_subset_of_shots(khalf,md,bigD):
    assert khalf in [0,1,2]
    rawShots=json.loads(bigD['shots_raw.JSON'][0])
    nShot=len(rawShots)
    nUse=nShot//2
    if khalf==1:
        rawShots=rawShots[:nUse]
    elif khalf==2:
        rawShots=rawShots[nUse:]
    else:
        rawShots=rawShots
        
    
    print('SSOS: use %d-ihalf=%d of %d shots'%(khalf,len(rawShots),nShot))
    state_counts = Counter()
    for state in rawShots: state_counts.update((state,))
    rawCounts=dict(state_counts)

    #.... opdate bigD & MD
    md['short_name']=md['short_name']+'_h%d'%khalf
    smd=md['submit']
    smd['num_shots']=len(rawShots)
    smd['use_half']=khalf

    bigD['counts_raw.JSON'][0]=json.dumps(rawCounts)
    nSol=len(rawCounts)
    md['job_qa']['num_sol']=nSol
    print("SSOS: numSol=%d"%(nSol))
    
    
#...!...!....................
def flatten_ranked_hexpatt_array(mshotV,hexpattV,nAtom):
    assert mshotV.shape==hexpattV.shape
    nSol=hexpattV.shape[0]
    inpL=[]    
    for k in range(nSol):
        mshot=mshotV[k]
        hexpatt=hexpattV[k].decode("utf-8")
        A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
        binpatt=A.bin
        inpL+=[binpatt]*mshot
    return inpL


#...!...!....................
def build_ranked_hexpatt_array(solCounter):
    nSol=len(solCounter) # number of distinct solutions (aka bitstrings)
    assert nSol>0  # fix it later
    rankedSol=solCounter.most_common(nSol)
    patt,mshot=rankedSol[0]
    nAtom=len(patt)
    
    NB=nSol # number of bitstrings,  this is sparse encoding 
    dataY=np.zeros(NB,dtype=np.int32)  # measured shots per pattern
    pattV=np.empty((NB), dtype='object')  # meas bitstrings as hex-strings (hexpatt)
    hammV=np.zeros(NB,dtype=np.int32)  # hamming weight of a pattern

    hexlen=int(np.ceil(nAtom / 4) * 4)
    pre0=hexlen-nAtom
    print('P4W: dataY:',dataY.shape, 'hexlen=%d, pre0=%d'%(hexlen,pre0))
    patt0='0'*pre0 # now pattern is divisible by 4
    for i in range(nSol):
        patt,mshot=rankedSol[i]
        # if 'rg' are used instead of '10' convert it to '1'0
        patt=patt.replace('g', '0').replace('r', '1')
        assert 'e' not in patt , "can't handle missing atoms at this stage"

        A=BitArray(bin=patt0+patt)
        hw=A.bin.count('1')  # Hamming weight
        if i<10: print('ii',i,patt,'hex=%s hw=%d mshot=%d'%(A.hex,hw,mshot))
        dataY[i]=mshot
        pattV[i]=A.hex
        hammV[i]=hw    
    return dataY,pattV,hammV,nAtom

