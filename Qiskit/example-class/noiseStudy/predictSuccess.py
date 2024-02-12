#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Given transpiled circuit and HW calibration 
Predict success rate for measured bit-string

'''

import time,os,sys
from pprint import pprint
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  write_yaml, read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path")
    parser.add_argument("--calibPath",default='hwcalib',help="input for HW calibration")
    
    parser.add_argument('--transpName', default='qft_4Q.q20b-Q4+2+8+3',
                        help=" transpiled circuit to be executed")
    parser.add_argument("--timeStamp",default='2019-10-01_10:27_UTC',
        help="can provide alternative date as string, e.g. 2019-09-28_16:11_UTC")

    args = parser.parse_args()
    args.prjName='predSucc_'+args.transpName
    args.dataPath+='/' 
    args.outPath+='/'
    args.calibPath+='/'
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def search_qubit_calib(q1,qubitCalL):
    #print(q1,len(qubitCalL))
    assert q1 <len(qubitCalL)
    x=qubitCalL[q1]
    [yT1,yT2]=x[:2]
    m1p0=x[5]
    
    assert yT2['name']=='T2'
    return yT1['value'],yT2['value'],m1p0['value']
    # measured1-prep0
    
#...!...!....................
def search_gate_calib(name,gateCalL):
    for g in gateCalL:
        if g['name'] != name: continue
        p=g['parameters']
        #print('eded',g,p)
        return [p[i]['value'] for i in range(2)]

#...!...!....................
def get_decoherence_err(gateCnt,qubitCnt,runTime,calibD,verb=0):
    usedQL=sorted(list(qubitCnt.keys()))
    #print('usedQubits=',usedQL)
    qubitCalL=calibD['qubits']

    errT1=0; errT2=0.; errM1P0=0
    for qid in usedQL:
        t1,t2,m1p0=search_qubit_calib(qid,qubitCalL)
        print('q:%d  T1=%.1f  T2=%.1f (uSec)'%(qid,t1,t2))
        #errT1+=runTime/t1
        #errT2+=runTime/t2
        errT1=max(errT1,runTime/t1)
        errT2=max(errT2,runTime/t2)
        errM1P0+=m1p0
    print('runT=%.3g (us), error due to T1=%.3g   T2=%.3g  ,  errM1P0=%.3g'%(runTime,errT1, errT2,errM1P0))
    return errT1, errT2, errM1P0

#...!...!....................
def countGates(circ,calibD,verb=1):
    #print('circOpt'); print(circ)
    dag = circuit_to_dag(circ)
    # dag.draw()  # will pop and stay
    graph_layers = dag.multigraph_layers()
    gateCalL=calibD['gates']
     
    cntG={}; cntQ={}
    totTime=0; lyrT=0
    totGateErr=0
    numSlowLyr=0
    for kLy,graph_layer in enumerate(graph_layers):
        op_nodes = [node for node in graph_layer if node.type == "op"]
        if verb>1:  print(kLy,'layer has num op:',len(op_nodes))
        lyrT=0
        for kNo, node in enumerate(op_nodes):
            if node.name=='barrier' : continue
            if node.name=='measure' : continue
            if node.name=='u1' : continue
            if len(node.qargs)==1:
                q1=node.qargs[0].index
                gateName='%s_%d'%(node.name,q1)
            else:
                q1=node.qargs[0].index
                q2=node.qargs[1].index
                gateName='%s%d_%d'%(node.name,q1,q2)
                if q2 not in cntQ : cntQ[q2]=0
                cntQ[q2]+=1
                
            if gateName not in cntG: cntG[gateName]=0
            cntG[gateName]+=1

            if q1 not in cntQ : cntQ[q1]=0
            cntQ[q1]+=1
            
            [gerr,glen]=search_gate_calib(gateName,gateCalL)
            totGateErr+=gerr
            lyrT=max(lyrT,glen)
            #print(kLy,'aa',gateName,glen,lyrT,totTime)
        # layer analyzed
        totTime+=lyrT
        if lyrT>0:  numSlowLyr+=1
        #print('bbb',lyrT, totTime)
    # graph analyzed
    print('cntG:',cntG)
    print('touched cntQ:',cntQ)
    print('totGateErr=%.3g'%totGateErr,' numSlowLyr=',numSlowLyr)
    return cntG, cntQ,totGateErr, totTime/1000.,numSlowLyr  # now tim is in muSec
        

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
if args.verb>2:
    print('base circOpt'); print(circOpt)
print('circD',circD.keys())
backName=circD['backend']
pprint(circD['info'])

inpF=args.calibPath+'/calibDB_%s_%s.yaml'%(backName,args.timeStamp)
print('read calibration from:',inpF)
hw_propertiesD=read_yaml(inpF)
dtStr=hw_propertiesD['last_update_date']
print('see keys:',hw_propertiesD.keys())


gateCnt,qubitCnt,errGate, runTime, numSlowLyr=countGates(circOpt,hw_propertiesD)
errT1,errT2,errM1P0=get_decoherence_err(gateCnt,qubitCnt,runTime,hw_propertiesD)
errD={'errTot': errGate+errT1+errT2+errM1P0,'errGate':errGate,'errT1':errT1,'errT2':errT2,'errM1P0':errM1P0,'runTime':runTime,'numSlowLyr':numSlowLyr}
print('\nEND-OK',errD)

