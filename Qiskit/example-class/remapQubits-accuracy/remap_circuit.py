#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
remap  transpiled circuit onto hardcoded  different qubits -hardcoded mapping

'''

import time,os,sys
from pprint import pprint
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import QuantumRegister
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  write_yaml, read_yaml, circ_2_yaml


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path")
    
    parser.add_argument('-t','--transpName', default='grover_3Qas01.q20b-Q1+2+3',
                        help=" transpiled circuit to be executed")

    args = parser.parse_args()
    args.prjName='subMitg_'+args.transpName
    args.dataPath+='/' 
    args.outPath+='/'
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def run_remap(circ0, srcTrgL, addInversion=False, verb=1):
    if len(srcTrgL)==0:  return circ0

    nqubits=len(circ0.qubits)
    # QuantumRegister needs to be instantiated to create the Layout
    qr = QuantumRegister(nqubits,'q')

    # Create the PassManager that will hold the passes we will use
    pass_manager = PassManager()

    # create full 1:1 mapping, to be modiffied as neede
    qrMap={ qr[i]:i for i in range(nqubits)}

    fullMap=[]
    if addInversion:
        for (a,b) in srcTrgL:
            fullMap.append((a,b))   
            fullMap.append((b,a))
    else:
        fullMap=srcTrgL

    print('RM: fullMap=',fullMap)
    for (a,b) in fullMap:
        qrMap[qr[a]]=b

    if verb>1: print(qrMap)
    layout = Layout(qrMap)

    # Create the passes that will remap your circuit to the layout specified
    set_layout = SetLayout(layout)
    apply_layout = ApplyLayout()

    # Add passes to the PassManager. (order matters, set_layout should be appended first)
    pass_manager.append(set_layout)
    pass_manager.append(apply_layout)

    # Execute the passes on your circuit
    remapped_circ = pass_manager.run(circ0)
    if verb>1:
        print(remapped_circ)
    return remapped_circ


# Import Aer
from qiskit import Aer
from qiskit import execute
import numpy as np
#...!...!....................
def compareUnitaries(circAr,backend):
    print('compUnit',backend)
    not_correc_indexing
    ua=np.zeros(1)
    for qc in circAr:
        print('see circ:',circ.name)
        print( '\n------- Unitary B')
        job = execute(qc, backend)
        result = job.result()
        ub=result.get_unitary(qc, decimals=3)
        # Show the results
        print('Ub=',ub.shape, type(ub))
        if ua.ndim==1: ua=ub
        print(qc)
    print('\nelement by element, i,j, diff,sum')

    n=ua.shape[0]
    for i in range(n):
        for j in range(n):
            a=ua[i,j]
            b=ub[i,j]
            d=a-b
            dm=(d*d.conjugate()).real
            if abs(dm)<0.1 : continue
            print(i,j,dm,a-b,a+b)

#...!...!....................
def runLocalSimulator(qc,backend):
    #print('locSim',backend)
    job_sim = execute(qc, backend, shots=1024)

    # Grab the results from the job.
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    print(' locSim counts=',counts)



#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

metaD={}
transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
transpD=read_yaml(transpF)
if args.verb>2:  print('M:  transpD'); pprint(transpD)

circD=transpD['circOpt']
qasmOpt=circD['qasm']


''' enable 2 lines to change backend
qasmOpt=qasmOpt.replace('qreg q[5]','qreg q[3]')
circD['backend']='ibmq_boeblingen'
#''' 

circOpt=QuantumCircuit.from_qasm_str(qasmOpt)
print('M:base circOpt'); print(circOpt)


# 2 cases:  
# a) I provide half o fmapping an autw-generate the inverse  OR
# b) I need to provide full mapping

remapPlan={}

'''  QFT-4Q merc-star : 6 possible mappings
addInv=False
# 3 rotations
remapPlan['Q16_mapA']=[]
remapPlan['Q16_mapB']=[(15,11),(11,17),(17,15)]
remapPlan['Q16_mapC']=[(15,17),(17,11),(11,15)]
# 3 reversed rotations
remapPlan['Q16_mapD']=[(11,17),(17,11)]
remapPlan['Q16_mapE']=[(11,15),(15,11)]
remapPlan['Q16_mapF']=[(17,15),(15,17)]
#'''

#1map16to6=[(16,6),(11,1),(15,5),(17,7)]

'''  play-2Q circuit
remapPlan['Q1_mapA']=[]
remapPlan['Q1_mapB']=[(0,2),(2,0)]
#'''

#'''  QFT-4Q  daisy-chain : 4 possible mappings
addInv=False
remapPlan['Q8_mapA']=[] # as-is
remapPlan['Q8_mapB']=[(1,8),(8,1),(2,3),(3,2)] # invert
remapPlan['Q3_mapA']=[(1,6),(2,1),(3,2),(8,3),(6,8)] # shift by 1
remapPlan['Q3_mapB']=[(1,3),(3,1),(8,6),(6,8)]  # shift by 1 & invert
#...

# run local simulator on each circuit
backend3 = Aer.get_backend('qasm_simulator')

xL=args.transpName.split('-Q'); #print('xL',xL)
coreName=xL[0]+'-'
expD={}
for qmapVer in remapPlan:
    srcTrgL=remapPlan[qmapVer]
    circ=run_remap(circOpt,srcTrgL, addInversion=addInv)
    #1circ=run_remap(circ,map16to6, addInversion=True); qmapVer=qmapVer.replace('Q16','Q6')
    
    circ.name=coreName+qmapVer
    print('ver=',qmapVer,circ.name)
    runLocalSimulator(circ,backend3)
    expD[qmapVer]=circ_2_yaml(circ,verb=0)
    expD[qmapVer]['optimLevel']='remapped'

print('M:aa',expD.keys())
transpF=args.outPath+'transp.'+coreName+'Q8-3_mapAB.yaml'

#XtranspD['backend']=circD['backend']
#XcircD.pop('backend') # tmp cleanup

transpD['circExp']=expD
write_yaml(transpD,transpF)
print(transpD.keys())

print('\nEND-OK')

# Run the quantum circuit on a unitary simulator backend
#backend2 = Aer.get_backend('unitary_simulator')
#compareUnitaries([circ,circOpt],backend2)

