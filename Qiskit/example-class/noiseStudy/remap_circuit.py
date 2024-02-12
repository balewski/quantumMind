#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Part 1') remap  transpiled circuit onto hardcoded  different qubits -hardcoded

'''

import time,os,sys
from pprint import pprint
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import QuantumRegister
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout
from NoiseStudy_Util import circ_2_yaml
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  write_yaml, read_yaml


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path")
    
    parser.add_argument('-t','--transpName', default='hidStr_2qm.ibmq_boeblingen-v10',
                        help=" transpiled circuit to be executed")

    args = parser.parse_args()
    args.prjName='subMitg_'+args.transpName
    args.dataPath+='/' 
    args.outPath+='/'
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

   
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
circOpt=QuantumCircuit.from_qasm_str(qasmOpt)
print('base circOpt'); print(circOpt)
nqubits=len(circOpt.qubits)
#nqubits=4
print('num qubits=',nqubits)

# QuantumRegister needs to be instantiated to create the Layout
qr = QuantumRegister(nqubits,'q')

# Create the PassManager that will hold the passes we will use
pass_manager = PassManager()

qrMap={ qr[i]:i for i in range(nqubits)}
#print(qrMap)
'''
qrMap[qr[1]]=3
qrMap[qr[3]]=1
qmapVer='-Q3+2+1'
'''
qrMap[qr[3]]=15
qrMap[qr[2]]=16
qrMap[qr[1]]=17
qrMap[qr[15]]=3
qrMap[qr[16]]=2
qrMap[qr[17]]=1
qmapVer='-Q17+16+15'

print(qrMap)
layout = Layout(qrMap)

# Create the passes that will remap your circuit to the layout specified above
set_layout = SetLayout(layout)
apply_layout = ApplyLayout()

# Add passes to the PassManager. (order matters, set_layout should be appended first)
pass_manager.append(set_layout)
pass_manager.append(apply_layout)

# Execute the passes on your circuit
remapped_circ = pass_manager.run(circOpt)
print(remapped_circ)

xL=args.transpName.split('-Q')
#print('xL',xL)
transpName2=xL[0]+qmapVer
transpF=args.outPath+'transp.'+transpName2+'.yaml'

name=circD['info']['name']

print(name,transpF)
transpD['circOpt']=circ_2_yaml(name,remapped_circ)
transpD['circOpt']['backend']=circD['backend']
transpD['circOpt']['optimLevel']=circD['optimLevel']
write_yaml(transpD,transpF)


print('\nEND-OK')

