#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 1) use Qiskit to trnspile (optimize, L=3)  circuit layout for given hardware and determine which qubits will be read out at the end.

required INPUT:
--circName: ghz_5qm
--backendName  abreviated=q20t
-- initLayout (optional)

The ideal circuit is read from
 data/ghz_5qm.qasm

The  Qiskit optimized layout using level-3 optimization for given device

'''

# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
from qiskit import QuantumCircuit, transpile 

from NoiseStudy_Util import circ_2_yaml
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml,circuit_summary


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='qcdata',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument('-s',"--rnd_seed",default=123,type=int,help="for transpiler")

    parser.add_argument("-c","--circName", default='ghz_5qm',
                        help="name of ideal  circuit")

    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument('-b','--backName',default='q5o',
                        choices=['q5o','q14m','q20p','q20b','q20t'],
                        help="backend for transpiler" )
    parser.add_argument('-Q',"--qubitLayout", nargs='+',type=int,default=None,
                        help="reqest HW map , list only used qubits ")


    args = parser.parse_args()
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

circF=args.dataPath+'/'+args.circName+'.qasm'
circOrg=QuantumCircuit.from_qasm_file( circF )
print('\ncirc original: %s, depth=%d'%( circF,circOrg.depth()))
print(circOrg)
metaD['circOrg']=circ_2_yaml(args.circName,circOrg)
#

# prepeare for transpilation
backend=access_backend(args.backName)

print('basis gates:', backend.configuration().basis_gates)
assert not backend.configuration().simulator

print(' Layout using optimization_level=',args.optimLevel)
circOpt = transpile(circOrg, backend=backend, optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed, initial_layout=args.qubitLayout) 
print(circOpt)

#print('circOpt measured qubits, ordered =',get_measeured_qubits(circOpt))

metaD['circOpt']=circ_2_yaml(args.circName+'+L%d'%args.optimLevel,circOpt)
meas_qubitL=metaD['circOpt']['info']['meas_qubit']
assert len(meas_qubitL)>0
qmapVer='Q%d'%meas_qubitL[0]
for q in meas_qubitL[1:]: qmapVer+='+%d'%q 

args.circName2=args.circName+'.%s-%s'%(args.backName,qmapVer)
args.prjName='transp_'+args.circName2

metaD['circOpt']['backend']=backend.name()
metaD['circOpt']['seed']=args.rnd_seed
metaD['circOpt']['optimLevel']=args.optimLevel

transpF=args.outPath+'/transp.'+args.circName2+'.yaml'
write_yaml(metaD,transpF)




