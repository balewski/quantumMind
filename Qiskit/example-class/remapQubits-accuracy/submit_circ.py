#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit job with mutiple clones of transpiled circuit

Required INPUT:
 --transpName : ghz_5qm
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
from qiskit.qasm import pi

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml, read_yaml, circuit_summary
from  ReadErrCorrTool  import ReadErrCorrTool

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument('-t','--transpName', default='grover_3Qas01.q20b-Q1+2+3',
                        help=" transpiled circuit to be executed")

    parser.add_argument( "--mock", action='store_true',
                         default=False, help="run on simulator instantly")
    parser.add_argument('-s','--shots',type=int,default=4096, help="shots")
    parser.add_argument('-r','--repeatCycle', type=int, default=30, help="number of copies of the circuit")
    parser.add_argument("--noErrCorr",dest='doReadErrCorr',action='store_false',
                         default=True, help="disable X-term for batch mode")

    
    args = parser.parse_args()
    args.prjName='subCircAr_'+args.transpName
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    return args

from qiskit.converters import circuit_to_dag, dag_to_circuit

#...!...!....................
def build_invCX_dag():
    q = QuantumRegister(2, 'p')
    qc = QuantumCircuit(q)
    qc.u2(0,pi,0)
    qc.u2(0,pi,1)
    qc.cx(0,1)
    qc.u2(0,pi,0)
    qc.u2(0,pi,1)    
    print(qc)
    return circuit_to_dag(qc),q

#...!...!....................
def force_unidirCX(qc0,trgL,verb=1):
    miniDag,qreg=build_invCX_dag()
    if verb>0:
        print('input circ'); print(qc0)
    dag=circuit_to_dag(qc0)
    gateL=dag.gate_nodes()
    print('\ngate_nodes: ',len(gateL))

    for node in gateL:
        if node.name!='cx': continue
        #print('nn',node.name)
        qL=[qr.index for qr in  node.qargs]
        qT=tuple(qL)
        if qT in trgL :
            print('keep cx ',qT)
            continue
        print('invert cx',qT)
        dag.substitute_node_with_dag(node=node, input_dag=miniDag, wires=[qreg[1],qreg[0]])

    qc3=dag_to_circuit(dag)
    if verb>0:
        print('final circuit qc3');    
        print(qc3)
    return qc3


#...!...!....................
def submitInfo_2_yaml(job_exe, args,expConf):  # packs job submission data for saving
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
    expInfo['conf']=expConf
    outD={'submInfo':submInfo, 'expInfo':expInfo}
    #pprint(outD)
    return outD


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

metaD={}
transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
blobD=read_yaml(transpF)

circD=blobD['circOpt']
if args.verb>1:  print('M: circOpt data:'); pprint(circD)
backName=blobD['backend']
backend=access_backend(backName)
print('backend=%s'%(backend))

meas_qubitL=[]

if 'circExp' in blobD:
    expD=blobD['circExp']
else:
    #OLD format: 1 transpiled circuit 
    qasmOpt=circD['qasm']
    circOpt=QuantumCircuit.from_qasm_str(qasmOpt)
    print('base circOpt name:',circOpt.name); print(circOpt)
    expD={'circOpt':circD} 
circTemplL=[]
expConf={}
for expId in expD:
    exp=expD[expId]
    circ=QuantumCircuit.from_qasm_str(exp['qasm'])
    circ.name=expId
    print('M: circ:',expId,exp['info']['name'])
    circTemplL.append(circ)        
    qcSum=circuit_summary(circ,verb=0)
    if len(meas_qubitL)==0: meas_qubitL=qcSum['meas_qubit']
    if len(circTemplL)==1:  pprint(qcSum)

    expConf[expId]={'meas_qubit' :qcSum['meas_qubit'], 'ancilla': qcSum['ancilla']}
    if args.doReadErrCorr: assert meas_qubitL==qcSum['meas_qubit']

#ok77
if args.doReadErrCorr:
    rdCorConf={'meas_qubit' :meas_qubitL,'nqubit':backend.configuration().n_qubits,'cal_name':'rderr'}
    rdCor=ReadErrCorrTool(rdCorConf)
    rdCorCircL=rdCor.make_circuits(verb=0)
    #rdCorCircL=rdCorCircL[0:1]
    numCyc=len(rdCorCircL)*1024//args.shots
    rdCorConf['num_cycles']=numCyc
    rdCorCircL*=numCyc  # replicate clibration circuit to reach target stat precicison
    print('total rdCorCirc count=',len(rdCorCircL),' using numCyc=',numCyc) 

'''
# add variants with uni-directional cx.
#circA=force_uniCX(circOpt,[(1,0),(2,1)],verb=0)

circA=force_unidirCX(circOpt,[(16,15),(16,17)],verb=0)
circA.name=qmapVer+'_cxA'
circB=force_unidirCX(circOpt,[(15,16),(17,16)],verb=0)
circB.name=qmapVer+'_cxB'

for x in [circA,circB]:
    circTemplL.append(x)
    nameTemplL.append(x.name)
    qcSum=circuit_summary(x.name,x,verb=0)
    pprint(qcSum)
'''

cycLen=len(circTemplL)
print('procured %d different circuits: '%cycLen,end='')
print(list(expConf.keys()))
    
print('Execute @ backend=%s, cycLen=%d x %d cycles, shots/state=%d'%(backend,cycLen,args.repeatCycle,args.shots))
# No transpiler is used

# replicate circuits as desired
circAr=[]
for i in range(args.repeatCycle):
    for circ in circTemplL:
        circAr.append( copy.deepcopy(circ))
print('circAr len=',len(circAr))
if args.doReadErrCorr:  circAr+=rdCorCircL
assert len(circAr)>0

'''
# HACK
qF='out/bad1.qasm'
circAr=[QuantumCircuit.from_qasm_file(qF)]

for circ in circAr:
    print('aa',circ.name)
    print(circ)
    print(circ.qasm())
'''

random.shuffle(circAr)

if args.mock:
    print('MOCK execution, no jobs submitted to device, quit')
    exit(33)

qobj_circ = assemble(circAr, backend, shots=args.shots)
job_exe =  backend.run(qobj_circ)

outD=submitInfo_2_yaml(job_exe, args, expConf)
print('\n *) Submitted calib:    %s  %s   numCirc=%d\n'%(args.prjName,outD['submInfo']['jid6'],len(circAr)),outD,'\n')

if args.doReadErrCorr: outD['rdCorInfo']=rdCorConf


outF=args.dataPath+'/submInfo-%s.yaml'%outD['submInfo']['jid6']
write_yaml(outD,outF)

print('\nEND-OK')
   
