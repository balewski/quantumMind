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
from networkx import nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

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
    parser.add_argument('-d',"--dataPath",default=None,help="data path if timeStamp is provided")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")

    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j','q53r','sim'],help="backend for transpiler" )
    parser.add_argument('-k',"--numDrop",type=int, default=1,
                       help="number of dropped items per observable")

    args = parser.parse_args()
    args.outPath+='/' 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    if args.timeStamp!=None: assert  args.dataPath!=None
    if args.dataPath!=None:
        assert os.path.exists(args.dataPath)
    return args


#...!...!....................
def ana_qubits_calib( calibD):
    # all observables are ' larger is better'
    valD={'T1':[],'T2':[],'rdFid':[]} # rdFid=1-readout_error
    
    for q,oneL in enumerate(calibD):
        #print(q,'oneL:', len(oneL)); pprint(oneL);ok99
        for k,i in [ ('T1',0), ('T2',1), ('rdFid',3) ]:
            val=oneL[i]['value']
            if i==3: val =1-val
            valD[k].append([q,val])
    #pprint(valD)
    dropL=[]
    for gt in valD:
        sortL = sorted(valD[gt], key=lambda kv: kv[1],reverse=False)
        #print(gt, 'sortL:',sortL)
        for i in range(args.numDrop):
            dropL.append(sortL[i][0])

    print('ana_qubits_calib end drop:',dropL)
    return set(dropL)

#...!...!....................
def ana_1Qgates_calib( calibD):
    ''' input structure 
        5 types of gates, analyze only  u2,u3
            each type can have up to Nq instances  for 1q and 
        output:[u2,u3]  [qid,'gate_error']
    '''
    valD={'u2':[],'u3':[]}  # 2D entries in the lists :[err,duration] 

    for recF in calibD:
        qL=recF['qubits']
        if len(qL)!=1 : continue # keep 1Q gates
        gType=recF['gate']
        if gType=='u1' : continue  # they are performed in software.  Thus, U1 gates are “free” in some sense.
        if gType=='id' : continue  #  I don't use them
        
        parL=recF.pop('parameters')

        valD[gType].append( [qL[0],parL[0]['value']])
    #pprint(valD)

    dropL=[]
    for gt in valD:
        sortL = sorted(valD[gt], key=lambda kv: kv[1],reverse=True)
        #print(gt, 'sortL:',sortL)
        for i in range(args.numDrop):
            dropL.append(sortL[i][0])

    print('ana_1Qgates_calib end drop:',dropL)
    return set(dropL)
    
    
#...!...!....................
def ana_2Qgates_calib(calibD):
    # input structure :        1 types of gates: cx
    dupL=[]
    valD2={}  # 2D entries in the lists :[err,duration]
    for recF in calibD:
        qL=recF['qubits']
        if len(qL)!=2 : continue # drop 1Q gates
        assert recF['gate']=='cx'
        if [qL[1],qL[0]] in dupL: continue # drop reverese indexed cx
        dupL.append(qL)
        parL=recF.pop('parameters')
        vL=[ rec['value'] for rec in parL]
        valD2[tuple(qL)]=vL       
    #pprint(valD2)

    dropL=[]
    #since both 'gate_error', 'gate_length' are 'smaller is better' we  flag the largest one
    for j in range(2):
        sortL = sorted(valD2.items(), key=lambda kv: kv[1][j],reverse=True)
        #print(j, 'sortL:',sortL)
        for i in range(args.numDrop):
            dropL.append(sortL[i][0])

    print('ana_2Qgates_calib end drop:',dropL)
    return set(dropL)

#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

backend=access_backend(args.backName)
args.prjName='pruneHW_.%s'%(backend.name())

plot=Plotter_HWcalib(args )

print('\nmy backend=',backend)
print(backend.status())
print('nqubit=',backend.configuration().n_qubits)
hw_config=backend.configuration().to_dict()
print('\nconfiguration :',hw_config.keys())
print('backend=',backend)

if args.verb>1:
    print('\ndump HW configuration')
    pprint(hw_config)
    
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
delT_h=(nowT2-updateT).seconds/3660.
print('\nproperties updated %.1f h ago'%delT_h,'on:',updateT,hw_proper.keys())

if args.verb>2:
    print('dump HW calibration (aka properties)')
    pprint(hw_proper)

hw_edges=hw_config['coupling_map']
#print('HW edges:',hw_edges)

tmp=[ '%d %d'%(a,b) for [a,b] in hw_edges]
G=nx.parse_edgelist(tmp, nodetype = int)
G0=G.copy()

drEdL=ana_2Qgates_calib(hw_proper['gates'])
g2Txt='worst CX er|t: '
for e in drEdL:
    G.remove_edge(e[0],e[1])
    g2Txt+='%d_%d, '%(e[0],e[1])

drNdL1=ana_1Qgates_calib(hw_proper['gates'])
drNdL2=ana_qubits_calib(hw_proper['qubits'])
g1Txt='worst U2|3: '
for n in drNdL1: g1Txt+='%d, '%(n)
qTxt='worst T1|2|rdEr: '
for n in drNdL2: qTxt+='%d, '%(n) 
drNdL2.update(drNdL1)
for n in drNdL2: G.remove_node(n)

# plotting
figId=3
fig=plt.figure(figId,facecolor='white', figsize=(10,4))
nrow,ncol=1,2
GL=[G0,G]
for i  in range(2):
   xG=GL[i]
   ax = plt.subplot(nrow,ncol,1+i)
   pos=graphviz_layout(xG)
   nx.draw(xG, with_labels=True,pos=pos,node_color = 'yellow')
   if i==0:  ax.set_title(backend.name()+' connectivity')
   if i==1:
       ax.set(title='calib_date:'+dtStr+' k=%d'%args.numDrop)
       x0=0.01
       ax.text(x0,0.01,g2Txt,transform=ax.transAxes,color='b')
       ax.text(x0,0.15,g1Txt,transform=ax.transAxes,color='b')
       ax.text(x0,0.08,qTxt,transform=ax.transAxes,color='b')
plt.savefig('myGraph.png')
print('plot.show')
plt.show()

    



