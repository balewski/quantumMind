#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 5) Submit target circuit  to device

Required INPUT:
 --transpName : ghz_5qm
 --noiseInjection:  [ num_cycl, inj_mult, theta_std]
 --mock (optional)

Reads:
  hw/sim-jobs/transp.ghz_5qm.v10.yaml

write  out/subInfo.yaml  - job submission summary, if using API

'''

import time,os,sys
from pprint import pprint
import qiskit 
from qiskit import QuantumCircuit, assemble, transpile 
from NoisyCircuitGenerator import NoisyDagGenerator
from qiskit.converters import circuit_to_dag, dag_to_circuit

from NoiseStudy_Util  import submitInfo_2_yaml
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, print_measFitter

import pytz
from datetime import datetime

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path")
    
    parser.add_argument('-t','--transpName', default='hidStr_2qm.ibmq_boeblingen-v10',
                        help=" transpiled circuit to be executed")

    parser.add_argument( "--mock", action='store_true',
                         default=False, help="run on local simulator instantly")
    parser.add_argument( "--simibm", action='store_true',
                         default=False, help="run on simulator at IBM as batch")
    parser.add_argument( "--simloc", action='store_true',
                         default=False, help="run on local simulator, interactively")
    parser.add_argument('-s','--shots',type=int,default=2048, help="shots")
    parser.add_argument('-n','--noiseModel', nargs='+',default=[2,1,0.3], help="custom circuit noise: [ num_cycles, inj_mult, theta_std ] ")

    args = parser.parse_args()
    args.prjName='subCircAr_'+args.transpName
    args.dataPath+='/' 
    args.outPath+='/'
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def assembly_circuit_array_fromDAG(baseCirc,noiseModel,verb=1): 
    startT = time.time()
    num_sets=int(noiseModel[0])
    inj_mult=int(noiseModel[1]) ; assert inj_mult<=1 #tmp
    theta_std=float(noiseModel[2])

    #.... GENERATE NEW CIRCUITS
    circL=[]
    circGener=NoisyDagGenerator(baseCirc,verb=verb)
    gateNameL=circGener.gateNameL
    
    for k in range(num_sets):
        if verb>0: print(' generate cycle ',k)
        circ=circGener.baseCircuit()
        #print(circ0)
        circL.append( circ) #  add base circuit for each set
        if inj_mult==0: continue
        
        for gateNm in gateNameL:
            verb=verb and (k==0)
            # I need decompose because I injected a custom gate
            circ=circGener.noisyCircuit( gateNm, theta_std,verb)
            circOpt=circ.decompose()
            if verb>0: print('M:',gateNm, 'noisy transpiled to:',circOpt)
            circL.append( circOpt)
            #break

    if verb>0:
        print('assembly_circuit_array done, circLen=%d, took %d sec'%(len(circL),time.time()-startT))

    return circL,circGener.expMeta


#...!...!....................
def run_large_local_simu_fromDAG(baseCirc,circD,backend):
    num_cycles=int(args.noiseModel[0])
    print('split into %d cycles'%num_cycles)
    args.noiseModel[0]='1'
    # must fake action of  yields_2_metaD() from retr_circ
    outL=[]
    outD=None
    
    startT = time.time()
    for ic in range(num_cycles):
        if ic%10==0: print('simu cycle=',ic,'elaT=%.1f min'%((time.time()-startT)/60.))
        verb=(ic==0)
        circAr,expMetaD=assembly_circuit_array_fromDAG(baseCirc,args.noiseModel,verb=verb)
        # Compile and run the Quantum circuit on a simulator backend
        qobj_circ = assemble(circAr, backend, shots=args.shots)
        jobAr=backend.run(qobj_circ)
        countsD=jobAr.result().get_counts(0)
        if verb: print('circ[0]  counts:',countsD)
        jobHeadD = jobAr.result().to_dict()
        jobResD=jobHeadD.pop('results')
        numExp=len(jobResD)
        if verb:
            print('see numExp=',numExp,'job header:'); pprint(jobHeadD)
        assert jobHeadD['success']
        if outD==None:
            outD=submitInfo_2_yaml(jobAr,args,circD, args.noiseModel, expMetaD)
            nclbit=len(circD['info']['meas_qubit'])
        # ....   harvest results
        for exp in jobResD:
            #print('gg'); pprint(exp); ok34
            counts=exp['data']['counts']
            counts2={ bin(int(x,16))[2:].zfill(nclbit):float(counts[x] ) for x in counts} # get rid of numpy
            expName=exp['header']['name']
            injGate=expMetaD[expName]['injGate']
            outL.append( { 'shots': exp['shots'], 'counts':counts2  , 'name':expName,'injGate':injGate})

    elaT=time.time() - startT
    # print(' timzeone-aware dt')
    nowT2=datetime.utcnow().replace(tzinfo=pytz.utc)
    #print(nowT2,'tmz=',nowT2.tzinfo )
    outD['retrInfo']={'num_cycles':num_cycles,'simu_elaT':elaT,'exec_date':'%s'%nowT2}
    
    #print('rrr'); pprint(outL)
    outD['yieldsMitg']=outL
    outD['calibInfo']='none-simloc'

    outF=args.outPath+'/yieldArray-%s.yaml'%outD['submInfo']['jid6']
    write_yaml(outD,outF)
    print('local simulation of %d cycles done, took %.1f min'%(num_cycles,elaT/60.))

   
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

metaD={}
transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
blobD=read_yaml(transpF)
if args.verb>2:  print('M:  transp blob'); pprint(blobD)

circD=blobD['circOpt']

if args.verb>1:  print('M: circOpt data:'); pprint(circD)

backName=circD['backend']
if args.simibm: backName='ibmq_qasm_simulator' # redicrect to IBM simulator
if args.simloc: backName='loc' # redicrect to local simulator
backend=access_backend(backName)

qasmOpt=circD['qasm']
circOpt=QuantumCircuit.from_qasm_str(qasmOpt)
print('base circOpt'); print(circOpt)
circuit_summary('circOpt',circOpt)

if args.simloc:
    run_large_local_simu_fromDAG(circOpt,circD,backend)
    exit(0) # all is saved already
else:
    circAr,expMetaD=assembly_circuit_array_fromDAG(circOpt,args.noiseModel,verb=args.verb)

if args.mock:
    print('mock submission, quit')
    print('expMetaD'); pprint(expMetaD)
    exit(33)

#pprint(expMeta)
numCirc=len(circAr)

print('\n*) Execute @ backend=%s, numCirc=%d shots/state=%d'%(backend,numCirc,args.shots),', noise:',args.noiseModel)
qobj_circ = assemble(circAr, backend, shots=args.shots)
#print('\nqobj_circ'); pprint(qobj_circ.to_dict())

job_exe =  backend.run(qobj_circ)
outD=submitInfo_2_yaml(job_exe,args,circD, args.noiseModel,expMetaD)
print('\n *) Submitted circ-array:   %s'%args.transpName, outD['submInfo']['jid6'],'\n',outD['submInfo'])
outF=args.dataPath+'/submInfo-%s.yaml'%outD['submInfo']['jid6']
write_yaml(outD,outF)

print('\nEND-OK')

