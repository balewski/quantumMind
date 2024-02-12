#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Display calibration of selected qubits of  a device
'''

# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
from dateutil.parser import parse as date_parse
from datetime import datetime
import pytz

from Plotter_HWcalib import Plotter_HWcalib

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml,read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument("--timeStamp",default=None,help="alternative date as string, e.g. 2019-09-30_16:11_UTC or PCT")
    parser.add_argument('-d',"--dataPath",default=None,help="data path is used if timeStamp is provided")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")

    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j','q53r','sim'],
                        help="backend for transpiler" )
    parser.add_argument('-Q',"--qubitList", nargs='+',type=int,
                        default=[2,3,4],help="selected  qubits to be porcessed ")

    args = parser.parse_args()
    args.outPath+='/' 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    if args.timeStamp!=None and  args.dataPath==None:
        args.dataPath='hwcalib/'
    if args.dataPath!=None:
        assert os.path.exists(args.dataPath)
    return args


#...!...!....................
def extract_qubits_calib(qL, calibD,updateT):
    valD={}; metaD={'Qid':[],'timeLag':[]}
    for q in qL:
        oneL=calibD[q]
        metaD['Qid'].append(q)
        for rec in oneL:
            dateStr=rec['date'] # is in UTC
            # dateStr.replace('+00:00','+12:00')
            lagT= updateT-date_parse(dateStr)
            metaD['timeLag'].append(lagT.total_seconds())
            val=rec['value']
            valName=rec['name']
            if valName not in valD:
                valD[valName]=[]
                metaD[valName]={'unit':rec['unit']}
            valD[valName].append(val)
    return valD, metaD
    
#...!...!....................
def extract_gates1Q_calib(inp_qL, calibD,updateT):
    ''' structure 
        5 types of gates: id, u2,u3
            each type can have up to Nq instances
    '''
    valD2={}  # 2D entries in the lists :[err,dur] 
    metaD={'gateType':[],'Qid':[],'timeLag':[]}

    qS=set(inp_qL)
    for recF in calibD:
        qL=recF['qubits']
        if len(qL)!=1 : continue # keep 1Q gates
        if  len (qS  & set(qL))==0 : continue
        # gate acts upon one of listed qubits
        gType=recF['gate']
        if gType=='u1' : continue  # they are performed in software.  Thus, U1 gates are “error free” in some sense.
        parL=recF.pop('parameters')
        if gType=='id': metaD['Qid'].append(qL[0])
        if gType not in valD2:
            valD2[gType]=[]
            if gType=='u2':                
                metaD['observable']=[rec['name'] for rec in parL]
                metaD['unit']=[ rec['unit']  for rec in parL]
        vL=[ rec['value'] for rec in parL]
        valD2[gType].append(vL)
        
        # track time lag for errors
        dateStr=parL[0]['date']
        dateStr.replace('+00:00','+12:00')
        lagT= updateT-date_parse(dateStr)
        metaD['timeLag'].append(lagT.total_seconds())

    return valD2, metaD
    
#...!...!....................
def extract_gates2Q_calib(inp_qL, calibD,updateT):
    # structure :        1 types of gates: cx
    
    valD2={'cx':[]}  # 2D entries in the lists :[err,dur] 
    metaD={'Qid2':[],'timeLag':[]}

    qS=set(inp_qL)
    dupL=[]
    for recF in calibD:
        qL=recF['qubits']
        if len(qL)!=2 : continue # drop 1Q gates
        if  not  set(qL).issubset(qS) : continue
        # gate acts upon one of listed qubits
        assert recF['gate']=='cx'
        # drop reverese indexed cxx
        if [qL[1],qL[0]] in dupL: continue
        dupL.append(qL)
        parL=recF.pop('parameters')
        if len(valD2['cx'])==0:
            uL=[   rec['unit']  for rec in parL]
            metaD['observable']=[rec['name'] for rec in parL]
            metaD['unit']=uL
        qLstr='%d_%d'%(qL[0],qL[1])
        metaD['Qid2'].append(qLstr)            
        vL=[ rec['value'] for rec in parL]
        valD2['cx'].append(vL)
        # track time lag for errors
        dateStr=parL[0]['date']
        dateStr.replace('+00:00','+12:00')
        lagT= updateT-date_parse(dateStr)
        metaD['timeLag'].append(lagT.total_seconds())

    return valD2, metaD


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
targ_qL=args.qubitList
mapVer='Q%d'%targ_qL[0]
for q in targ_qL[1:]: mapVer+='+%d'%q 

backend=access_backend(args.backName)
args.prjName='hwcal_.%s-%s'%(backend.name(),mapVer)

plot=Plotter_HWcalib(args )

print('\nmy backend=',backend)
print(backend.status())
print('nqubit=',backend.configuration().n_qubits)
hw_config=backend.configuration().to_dict()
print('\nconfiguration :',hw_config.keys())
print('backend=',backend)
for x in [ 'max_experiments',  'max_shots', 'n_qubits' ]:
    print( x,hw_config[x])

if args.qubitList==[-1]:
    args.qubitList=[x for x in range(hw_config['n_qubits' ])]

if args.verb>1:
    print('\ndump HW configuration')
    pprint(hw_config)
    
''' use following info
*) hw_config['coupling_map'] ---> draw graph
'''

if args.timeStamp==None:
    hw_proper=backend.properties().to_dict()
    dtStr=hw_proper['last_update_date']
    print('show live calibration, posted at:',dtStr)
else:
    inpF=args.dataPath+'/calibDB_%s_%s.yaml'%(backend,args.timeStamp)
    #print('read calibration from:',inpF)
    hw_proper=read_yaml(inpF)
    dtStr=hw_proper['last_update_date']
    print('see keys:',hw_proper.keys())

    
updateT = date_parse(dtStr)
print('updateT',updateT,'tmz=',updateT.tzinfo)
nowT2=datetime.utcnow().replace(tzinfo=pytz.utc)
delT_h=(nowT2-updateT).total_seconds()/3660.
print('\nproperties updated %.1f h ago'%delT_h,'on:',updateT,hw_proper.keys())


if args.verb>2:
    print('dump HW calibration (aka properties)')
    pprint(hw_proper)

dtStr2=dtStr[:-6].replace('T',' ')+'_UTC' # replace sec+00 --> UTC
metaD={'calib_date':dtStr2, 'backend':'%s'%backend}
qvalD, qmetaD=extract_qubits_calib(args.qubitList, hw_proper['qubits'],nowT2)
metaD['qubits']=qmetaD

gval1qD, g1qmetaD=extract_gates1Q_calib(args.qubitList, hw_proper['gates'],updateT)
metaD['gates1Q']=g1qmetaD
#print('M: gval1qD',gval1qD)

gval2qD, g2qmetaD=extract_gates2Q_calib(args.qubitList, hw_proper['gates'],updateT)
metaD['gates2Q']=g2qmetaD
#print('M: gval2qD',gval2qD)
#print('M: meta2qD',g2qmetaD)

plot.qubits_calib( qvalD, metaD, figId=10)
axAr=plot.gates1Q_calib( gval1qD, metaD, figId=11)
plot.gates2Q_calib( gval2qD, metaD, axAr)

plot.display_all()



