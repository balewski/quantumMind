#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

 Explore alternative mappings of the initial circuit

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

import RelRank

sys.path.append(os.path.abspath("../noiseStudy/"))
from NoiseStudy_Util import circ_2_yaml

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml, read_yaml,circuit_summary


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='qcdata',help="input for everything")
    parser.add_argument("--calibPath",default='hwcalib',help="input for HW calibration")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument('-s',"--rnd_seed",default=123,type=int,help="for transpiler")

    parser.add_argument("-c","--circName", default='grover_3Qs00z',
                        help="name of ideal  circuit")

    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j'],
                        help="backend for transpiler" )
    parser.add_argument("--timeStamp",required=True,help="must provide date as string, e.g. 2019-09-28_16:11-UTC")

   
    parser.add_argument('-Q',"--qubitLayout", nargs='+',type=int,default=None,
                        help="suggest qubits to be used ")

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
print('\ncirc original: %s, depth=%d'%( circF,circOrg.depth()))
print(circOrg)

metaD['circOrg']=circ_2_yaml(args.circName,circOrg)
#

# code below only agregates gates
from qiskit.compiler import transpile
optLev=1
print('\n Transpile=(Optimize and fuse gates)  optLev=',optLev)
circBasic = transpile(circOrg, basis_gates=['u3','cx'], optimization_level=optLev)
print('\n only Decompose() qc')
print(circBasic)
pprint(circuit_summary('circBasic',circBasic))

# prepeare for transpilation
backend=access_backend(args.backName)

inpF=args.calibPath+'/calibDB_%s_%s.yaml'%(backend,args.timeStamp)
print('read calibration from:',inpF)
hw_propertiesD=read_yaml(inpF)
dtStr=hw_propertiesD['last_update_date']
#print('see keys:',hw_propertiesD.keys())
hw_properties=BackendProperties.from_dict(hw_propertiesD)

print('basis gates:', backend.configuration().basis_gates)
assert not backend.configuration().simulator

print(' Layout using optimization_level=',args.optimLevel,backend)

md5S=set()
for seed in range(100,990):
    if len(md5S)>=40: break
    circOpt = transpile(circOrg, backend=backend, backend_properties=hw_properties, optimization_level=args.optimLevel, seed_transpiler=seed, initial_layout=args.qubitLayout)

    ciya=circ_2_yaml('seed:%d'%seed,circOpt,verb=0)
    cisu=ciya['info']
    if cisu['gates'] >=65: continue
    md5=ciya['md5']
    if md5 in md5S : continue
    #if cisu['operations']['cx']!=15 : continue #tmp
    #if cisu['operations']['u3']!=8 : continue #tmp
    md5S.add(md5)
    #print('\n\n   -------------',seed);    print(circOpt)
    for x in ['barrier','measure']:
        del cisu['operations'][x]

    meas_qubitL=cisu['meas_qubit']
    print(len(md5S),cisu['name'],cisu['operations'],' gates:',cisu['gates'],'factors:',cisu['factors'],' measure:', meas_qubitL,end='' )#, 'md5:',md5[-10:])
    assert len(meas_qubitL)>0
    assert len(cisu['ancilla'])==0

    # compute rank of this circuit
    gateCnt,qubitCnt,errGate, runTime, numSlowLyr=RelRank.countGates(circOpt,hw_propertiesD,verb=0)
    errT1,errT2,errM1P0=RelRank.get_decoherence_err(gateCnt,qubitCnt,runTime,hw_propertiesD,verb=0)
    errD={'errTot': errGate+errT1+errT2+errM1P0,'errGate':errGate,'errT1':errT1,'errT2':errT2,'errM1P0':errM1P0,'runTime':runTime,'numSlowLyr':numSlowLyr}
    print('   RelRank=%.3f   g:%.3f  T1:%.3f  T2:%.3f'%(errD['errTot'],errD['errGate'],errD['errT1'],errD['errT2']))

    #print(circOpt) # prints whole circuit
    
    qmapVer='Q%d'%(meas_qubitL[0])
    for q in meas_qubitL[1:]: qmapVer+='-%d'%q
    qmapVer+='_s%d'%seed

    metaD['circOpt']=ciya

    args.circName2=args.circName+'.%s_%s'%(args.backName,qmapVer)
    args.prjName='transp_'+args.circName2

    metaD['circOpt']['calib_date']=dtStr
    metaD['circOpt']['backend']=backend.name()
    metaD['circOpt']['seed']=seed
    metaD['circOpt']['optimLevel']=args.optimLevel

    transpF=args.outPath+'/transp.'+args.circName2+'.yaml'
    write_yaml(metaD,transpF,verb=0)
    
    




