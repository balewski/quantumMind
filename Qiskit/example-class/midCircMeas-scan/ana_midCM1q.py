#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze   mid-circuit measurement experiment for 1 qubit

- no graphics
'''

import os
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5

from pprint import pprint
import numpy as np
from bitstring import BitArray


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QPIXL_dataVault")
                        
    parser.add_argument('-e',"--expName",  default='fexp_94e331',help='IBMQ experiment name assigned during submission')

    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QCloud_dataVault']
    args.dataPath=os.path.join(args.basePath,'meas')
    args.outPath=os.path.join(args.basePath,'ana')
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args

#...!...!....................
def verify_1q(countsAll,md):
    nameL=md['circ_sum']['circ_name']
    backName=md['submit']['backend']
    shots=md['submit']['num_shots']
    classN=md['payload']['class_name']
    print('\nEval %s on %s for %d shots'%(classN,backName,shots))
    totOK=True
    fracThr=0.2

    sumD={'q%d'%q:{} for q in range(md['submit']['tot_qubits']) }
    
    for name,cntV in zip(nameL,countsAll):
        circN,qid=name.split('_')
        if 'MXM'==circN:
            m1=(0,1);  m0=(1, 0)            
        if 'MCM'==circN:
            m1=(1,1);  m0=(1, 0)

        #print(name,'a',cntV,cntV.shape, cntV[3,0])
        
        n0=cntV[m0]
        n1=cntV[m1]

        navr=shots/2
        print('verify:%s n0=%d n1=%d navr=%d'%( name,n0,n1,navr))
        #print('counts',name, cntV)
        
        c1= (shots -n0 -n1)/navr
        c2= abs(n0 -n1) /navr
        isOK=c1 <fracThr and c2 <fracThr    
        msg='    PASS    ' if isOK  else '   ***FAILED***'
        print('  c1=%.2f, c2=%.2f'%(c1,c2),msg,name)
        totOK*=isOK
        sumD[qid][circN]=isOK
    msg='    PASS    ' if totOK  else '   ***FAILED***'
    print('Verify:',msg,' shots=%d, fracThr=%d, exp=%s\n'%(shots,fracThr,md['short_name']))
    print('\nSummary mid-circ-meas scan ',md['submit']['backend'],md['submit']['date'])
    pprint(sumD)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=2)
                    
    inpF=args.expName+'.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            for x in  expD:
                if 'qasm3' in x : continue
                print(x,expD[x])
        if args.verb>=4:
            #rec2=expD['circ_qasm3'][1].decode("utf-8")
            rec2=expD['transp_qasm3'][1].decode("utf-8") 
            print('\nM:qasm3 circ1:\n',rec2)

        stop3

  
    
    verify_1q(expD['counts_raw'],expMD)
    
    print('M:done')
