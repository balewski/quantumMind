#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Analyze   TomoU3  experiment

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
def ana_exp(countsAll,md):
    nameL=md['circ_sum']['circ_name']
    backName=md['submit']['backend']
    shots=md['submit']['num_shots']
    classN=md['payload']['class_name']
    print('\nEval %s on %s for %d shots'%(classN,backName,shots))
    evD={'q%d'%q:{} for q in range(md['submit']['tot_qubits']) }
    
    for name,cntV in zip(nameL,countsAll):
        print('circ:%s, counts:'%name,cntV)
        axMeas,_,qid=name.split('_')
        n0,n1=cntV
        ns=n0+n1
        prob=n1/(n0+n1)
        n0=max(1,n0)
        n1=max(1,n1)
        probEr=np.sqrt(n0*n1/ns)/ns
        ev=1-2*prob
        evEr=2*probEr
        
        evD[qid][axMeas]=(ev,evEr)
        #print(axMeas,'n0,n1',n0,n1)
    return evD

#...!...!....................
def verify(evD,md):
    backName=md['submit']['backend']
    shots=md['submit']['num_shots']

    phi=md['payload']['phi']
    theta=md['payload']['theta']
   
    cth=np.cos(theta)
    sth=np.sin(theta)
    cphi=np.cos(phi)
    sphi=np.sin(phi)

    uz=cth
    ux=sth*cphi
    uy=sth*sphi

    evT={'z':uz,'y':uy,'x':ux}
    #1print('True: ',evT)
    nSigThr=8
    print('\nCompare expected values on %s for 3 tomo-axis'%backName,md['submit']['date'])

    for q in range(md['submit']['tot_qubits']):
        isOK=True
        qid='q%d'%q
    
        for ax in list('xyz'):
            evM,evE=evD[qid][ax]
            dev=evM-evT[ax]
            nSig=abs(dev)/evE
            isOK*= nSig<nSigThr
            print('ax=%c  nSig=%.1f  true=%.3f, meas=%.3f +/- %.3f'%(ax,nSig,evT[ax],evM,evE) )
            
        msg='    PASS    ' if isOK  else '   ***FAILED***'
        print('verify:',qid,msg,' shots=%d per ax, nSigThr=%d exp=%s\n'%(shots,nSigThr,md['short_name']))

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

    evD=ana_exp(expD['counts_raw'],expMD)
    #1print('EV',evD)
    verify(evD,expMD)

    print('M:done')
