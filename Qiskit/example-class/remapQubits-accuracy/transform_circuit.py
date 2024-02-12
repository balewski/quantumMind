#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 0)  Explore alternative mappings of the initial circuit

required INPUT:
--circName: ghz_5qm
--backendName  abreviated=q20t
-- initLayout (optional)

The ideal circuit is read from
 qcdata/ghz_5qm.qasm

The  Qiskit optimized layout using level-3 optimization for given device

'''

# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
import pytz
from datetime import datetime
import random

from qiskit import QuantumCircuit, transpile
from qiskit.providers.models import BackendProperties

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml, read_yaml,circuit_summary, circ_2_yaml


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='qcdata',help="input for everything")
    parser.add_argument("--calibPath",default='hwcalib',help="input for HW calibration")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")

    parser.add_argument("-c","--circName", default='qft_4Qz',
                        help="name of ideal  circuit")

    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j'],
                        help="backend for transpiler" )
    parser.add_argument("--timeStamp",default=None,help="can provide alternative date as string, e.g. 2019-09-28_16:11-UTC")

   
    parser.add_argument('-Q',"--qubitLayout", nargs='+',type=int,default=None,
                        help="reqest HW map , list only used qubits ")


    args = parser.parse_args()
    args.dataPath+='/' 
    args.outPath+='/' 
    args.calibPath+='/' 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    for xx in [args.dataPath, args.outPath, args.calibPath]:
        assert  os.path.exists(xx)
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
print('\ncirc original: %s, depth=%d, name=%s'%( circF,circOrg.depth(),circOrg.name))
circOrg.name=args.circName
print(circOrg)
metaD['circOrg']=circ_2_yaml(circOrg)

# code below only agregates gates
from qiskit.compiler import transpile
optLev=1
print('\n Transpile=(Optimize and fuse gates)  optLev=',optLev)
circBasic = transpile(circOrg, basis_gates=['u3','cx'], optimization_level=optLev)
print('\n only Decompose() qc')
circBasic.name='circBasic'
print(circBasic)
pprint(circuit_summary(circBasic))


# prepeare for transpilation
backend=access_backend(args.backName)

print('basis gates:', backend.configuration().basis_gates)
assert not backend.configuration().simulator

print(' Layout using optimization_level=',args.optimLevel,backend)

if args.timeStamp==None:
    hw_proper=backend.properties().to_dict()
    dtStr=hw_proper['last_update_date']
    print('use live calibration, posted at:',dtStr)
    hw_properties=None
else:
    inpF=args.calibPath+'/calibDB_%s_%s.yaml'%(backend,args.timeStamp)
    print('read calibration from:',inpF)
    hw_propertiesD=read_yaml(inpF)
    dtStr=hw_propertiesD['last_update_date']
    #print('see keys:',hw_propertiesD.keys())
    hw_properties=BackendProperties.from_dict(hw_propertiesD)

md5S=set()
for seed in range(100,990):
    if len(md5S)>=40: break
    #if q1%5 not in [1,2,3] :continue
    circOpt = transpile(circOrg, backend=backend, backend_properties=hw_properties, optimization_level=args.optimLevel, seed_transpiler=seed, initial_layout=args.qubitLayout)

    circOpt.name='seed:%d'%seed
    ciya=circ_2_yaml(circOpt,verb=0)
    cisu=ciya['info']
    if cisu['gates'] >=60: continue
    md5=ciya['md5']
    if md5 in md5S : continue
    #if cisu['operations']['cx']!=15 : continue #tmp
    #if cisu['operations']['u3']!=8 : continue #tmp
    md5S.add(md5)
    #print('\n\n   -------------',seed);    print(circOpt)
    for x in ['barrier','measure']:
        del cisu['operations'][x]

    meas_qubitL=cisu['meas_qubit']
    ciop=cisu['operations']
    
    print(len(md5S),cisu['name'],', gates:',cisu['gates'],', factors:',cisu['factors'],', cx:',ciop['cx'], ', u3:',ciop['u3'], ', u2:',ciop['u2'], ', meas:', meas_qubitL,', ancilla:',cisu['ancilla'] )#, 'md5:',md5[-10:])

    
    print(circOpt) # prints whole circuit
    
    assert len(meas_qubitL)>0
    qmapVer='Q%d'%(meas_qubitL[0])
    for q in meas_qubitL[1:]: qmapVer+='+%d'%q
    #qmapVer='s%d'+qmapVer

    metaD['circOpt']=ciya

    args.circName2=args.circName+'.%s-%s'%(args.backName,qmapVer)
    args.prjName='transp_'+args.circName2

    metaD['circOpt']['calib_date']=dtStr
    metaD['backend']=backend.name()
    metaD['circOpt']['seed']=seed
    metaD['circOpt']['optimLevel']=args.optimLevel

    transpF=args.outPath+'/transp.'+args.circName2+'.yaml%s'%seed
    write_yaml(metaD,transpF,verb=0)

    




