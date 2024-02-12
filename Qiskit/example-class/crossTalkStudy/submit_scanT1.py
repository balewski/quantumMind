#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit jobs allowing determination T1 for 1 qubit 

Required INPUT:
 --qubit:
 --ancillaList:  to introduce delay
 --backName
 --mock (optional)
 
Prints/write  out/subInfo-jNNNNN.yaml  - job submission summary, fixed name

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import copy
import datetime
import random

from qiskit import assemble, QuantumRegister, ClassicalRegister, execute, Aer, QuantumCircuit

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml, read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument( "--sim", action='store_true',
                         default=False, help="run on simulator at IBM")
    parser.add_argument('-Q',"--qubitList", type=int, nargs='+', default=[2],
                        help="investigated qubit")
    parser.add_argument('-A',"--ancillaList", nargs='+',type=int,default=[14],
                        help="a pair of ancilla qubits for delays ")
    parser.add_argument( "--mock", action='store_true',
                         default=False, help="run on simulator instantly")
    parser.add_argument('-s','--shots',type=int,default=4096, help="shots")
    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j'],
                        help="backend for transpiler" )
    parser.add_argument('-r','--repeatCycle', type=int, default=2, help="number of copies of the circuit")
    parser.add_argument('--delayList', nargs='+', type=int, default=[0,1,2,4,8,16], help="list of dalayes of readout delays in uSec")
    
    args = parser.parse_args()
    qmapVer='Q%d'%args.qubitList[0]
    qmapVer+='_nT%d'%len(args.delayList)
    args.prjName='scan_'+qmapVer

    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    return args

#...!...!....................
def build_T1_circ(prepState,delayT_usec,nqubit,args):
    par_u3_duration=71./1000  # in uSec
    numTicks=int(delayT_usec/par_u3_duration)
    qid=args.qubitList[0] # tested qubit
    assert qid<nqubit
    assert len(args.qubitList)==1
    #print('xxx',delayT_usec,numTicks)
    name='scan_Q%d_us%d_pr%d'%(qid,delayT_usec,prepState)
    qr = QuantumRegister(nqubit,'q')
    cr = ClassicalRegister(1,'c')
    qc = QuantumCircuit(qr,cr,name=name)

    if prepState>0:
        qc.x(qid) # set initial state to 1
    qc.barrier()
    if numTicks>0:    # add delay as requested
        p1=args.ancillaList
        assert p1!=qid
        for ic in range(numTicks):
            qc.u3(0.1,0.2,0.3,p1)
        qc.barrier()
    qc.measure(qr[qid],cr[0])
    print('qc',qc.name); print(qc)    
    return qc


#...!...!....................
def submitInfo_2_yaml(job_exe, args,circD):  # packs job submission data for saving
    nowT = datetime.datetime.now()
    submInfo={}
    submInfo['job_id']=job_exe.job_id()
    submInfo['jid6']='j'+submInfo['job_id'][-6:] # the last 6 chars in job id , as handy job identiffier
    submInfo['submitDate']=nowT.strftime("%Y-%m-%d %H:%M")
    submInfo['backName']='%s'%job_exe.backend().name()
    submInfo['prjName']=args.prjName
 
    expInfo={}
    expInfo['shots']=args.shots
    expInfo['repeat_cycle']=args.repeatCycle
    expInfo.update(circD)
    outD={'submInfo':submInfo, 'expInfo':expInfo}
    #pprint(outD)
    return outD


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

backend=access_backend(args.backName)
print('backend=%s'%(backend))

hw_config=backend.configuration().to_dict()
print('\nconfiguration :',hw_config.keys())
print('backend=',backend)
for x in [ 'max_experiments',  'max_shots', 'n_qubits' ]:
    print( x,hw_config[x])

nqubit=hw_config['n_qubits']

# Generate one cycle of calibration circuits=
calCircL=[]
circNameL=[]
for delayT_usec in args.delayList:
    for prepState in [0,1]:
        circ= build_T1_circ(prepState,delayT_usec,nqubit,args)
        calCircL.append(circ )
        circNameL.append(circ.name)
cycLen=len(calCircL)
print('procured %d different circuits'%cycLen)
print(circNameL)

iexp=cycLen//3;  circ1=calCircL[iexp]
print('example calib circuit iexp=%d of %d'%(iexp,cycLen))
print(circ1)

print('Execute @ backend=%s, cycLen=%d x %d cycles, shots/state=%d'%(backend,cycLen,args.repeatCycle,args.shots))
# No transpiler is used

# replicate circuits as desired
circAr=[]
for i in range(args.repeatCycle):
    for circ in calCircL:
        circAr.append( copy.deepcopy(circ))
print('circAr len=',len(circAr))
random.shuffle(circAr)

if args.mock:
    print('MOCK execution, no jobs submitted to device, quit')
    exit(33)

qobj_circ = assemble(circAr, backend, shots=args.shots)
job_exe =  backend.run(qobj_circ)

circD={ 'meas_qubit' :args.qubitList,'ancilla':args.ancillaList,'exp_name':circNameL, 'delayT_usec':args.delayList}

outD=submitInfo_2_yaml(job_exe,args ,circD)
print('\n *) Submitted calib:    %s  %s   numCirc=%d\n'%(args.prjName,outD['submInfo']['jid6'],len(circAr)),outD,'\n')

outF=args.dataPath+'/submInfo-%s.yaml'%outD['submInfo']['jid6']
write_yaml(outD,outF)

print('\nEND-OK')
   
