#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Submit job with mutiple  RB circuits
no read-err-corr

Required INPUT:
 --mock (optional)
 
Prints/write  out/subInfo-jNNNNN.yaml  - job submission summary, fixed name

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import copy
import datetime
import random

from qiskit import assemble, QuantumRegister, ClassicalRegister, execute, Aer, QuantumCircuit, transpile
#from qiskit.qasm import pi
#Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb


sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml, read_yaml, circuit_summary

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument('-Q',"--qubitLayout", nargs='+',type=int,default=None,
                        help="define which qubits to be tested ")

    parser.add_argument( "--mock", action='store_true',
                         default=False, help="run on simulator instantly")
    parser.add_argument('-s','--shots',type=int,default=1024, help="shots")
    parser.add_argument('-b','--backName',default='q5s',
                        help="backend for execution" )
    parser.add_argument('-L','--optimLevel',type=int,default=1, help="transpiler ")
     
    args = parser.parse_args()
    args.rnd_seed=111 # for transpiler
    args.prjName='subRB'
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    return args

from qiskit.converters import circuit_to_dag, dag_to_circuit


#...!...!....................
def submitInfo_2_yaml(jobL, args,expInfo):  # packs job submission data for saving
    nowT = datetime.datetime.now()
    submInfo={}
    submInfo['job_id']=[job.job_id() for job in jobL]
    submInfo['jid6']=['j'+x[-6:] for x in submInfo['job_id']] # the last 6 chars in job id , as handy job identiffier
    submInfo['submitDate']=nowT.strftime("%Y-%m-%d %H:%M")
    submInfo['backName']='%s'%job_exe.backend().name()
    submInfo['prjName']=args.prjName
 
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

#..... Generate RB circuits....
rb_opts = {}

rb_opts['length_vector'] = [1, 2, 4, 8,16, 24,32] #Num. of Cliffords in the sequence
rb_opts['nseeds'] = 30 #Number of random sequences of Cliffords 

#Default pattern of  tested qubits
rb_opts['rb_pattern'] = [[1,2]]  # includes cx
#rb_opts['rb_pattern'] = [[0],[1]]  # only 1Q-gates
rb_opts['align_cliffs']=True # adds barieres

rb_circs, xdata = rb.randomized_benchmarking_seq(**rb_opts)
#As an example, we print the circuit corresponding to the first RB sequence
print(rb_circs[0][0])

print('A-len',len(rb_circs))
for x in rb_circs:
    print('B-len',len(x))
    for circ in x:
        print(circ.name,', depth:',circ.depth(),', size:',circ.size())
print('xdata:',xdata)  #the Clifford lengths

backend=access_backend(args.backName)
print('backend=%s'%(backend))
basis_gates = ['u1','u2','u3','cx'] 

jobL=[]
qobjL = []
for rb_seed,rb_circL in enumerate(rb_circs):
    print('Compiling seed %d, num. circ=%d'%(rb_seed,len(rb_circL)))
    rb_circT = transpile(rb_circL, basis_gates=basis_gates, optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed)

    print('Execute @ backend=%s, numCirc=%d , shots/state=%d, rb_seed=%d ...'%(backend,len(rb_circT),args.shots,rb_seed))
    
    # No transpiler is used
    qobj_asm = assemble(rb_circT, backend, shots=args.shots)
    qobjL.append(qobj_asm)

    if args.mock:
        print('  MOCK execution, no jobs submitted to device, continue')
    else:
        job_exe =  backend.run(qobj_asm)
        jobL.append(job_exe)


print('\n Count the number of single and 2Q gates in the 2Q Cliffords')
# separately for each set of qubits
gpCfL=[]
for j,qset_pattern in enumerate(rb_opts['rb_pattern']):
    gates_per_cliff = rb.rb_utils.gates_per_clifford(qobjL, xdata[j],basis_gates, qset_pattern)
    gpCfL.append(gates_per_cliff)
    print('qset=%d, Q:'%(j),qset_pattern,' gpCf shape:',gates_per_cliff.shape,' gates per Clifford')
    for k in range(gates_per_cliff.shape[0]):
        print('  Q=%d : '%qset_pattern[k],end='')
        print([ (a,'%.3f'%b) for a,b in zip(basis_gates ,gates_per_cliff[k])])
    
qcSum=circuit_summary(circ,verb=0)        
expInfo={'shots':args.shots,'rb_xdata':xdata, 'rb_pattern':rb_opts['rb_pattern'],'meas_qubit' :qcSum['meas_qubit'], 'ancilla': qcSum['ancilla'] }
expInfo['gates_per_cliff']=gpCfL
expInfo['basis_gates']=basis_gates
pprint(qcSum)

outD=submitInfo_2_yaml(jobL, args, expInfo)
jid6=outD['submInfo']['jid6'][-1]+'_x%d'%rb_opts['nseeds']
print('\n *) Submitted RB: %s  numCirc=%d x %d seeds for jid6: %s\n'%(args.prjName,len(rb_circT),rb_opts['nseeds'],jid6),' '.join(outD['submInfo']['jid6']))
pprint(outD)

outF=args.dataPath+'/submInfo-%s.yaml'%jid6
write_yaml(outD,outF)

print('\nEND-OK')
   
