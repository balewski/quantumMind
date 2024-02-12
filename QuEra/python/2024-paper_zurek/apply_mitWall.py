#!/usr/bin/env python3
""" 
 Kibble-Zurek SPAm mitigation
 Apply   matrix inversion

- no graphics
"""

__author__ = "Jan Balewski, Milan Kornjaca"
__email__ = "janstar1122@gmail.com, Milan Kornjaca"

import numpy as np
import  time
import sys,os
from pprint import pprint
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.UAwsQuEra_job import flatten_ranked_hexpatt_array, build_ranked_hexpatt_array

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--dataPath",default='4milo',help="input location")

    parser.add_argument('-e',"--expName",  default='ideal_o58pf0.05.zneWall.h5',help=' name of analyzed  experiment ')

    args = parser.parse_args()
  
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.dataPath)    

    return args


#function to generate 2x2 confusion matrix inverse from detection infidelities, basis {00, 01, 10, 11}
#...!...!....................
def conf_matrix_inverse2x2(eps01, eps10):
    Ainv=np.linalg.inv(np.array([[1-eps01,eps10],[eps01, 1-eps10]]))
    return np.kron(Ainv,Ainv)

#print(np.round(conf_matrix_inverse2x2(0.01, 0.08),2))


#Calculates average number of domain walls with error and mitigated results with error from input array of bistrings.
#...!...!....................
def wall_count_mitig_v1(bitpattL, eps01=0.01, eps10=0.08):
    """Calculates average number of domain walls with error and mitigated results with error from input array of bistrings, ``bitstring data``.
    
    Parameters
    ----------
    ``bitpattL`` : list of measured bitstrings encoded as 0s and 1s
        length of one bitstring is system size, number of bitstrings is the number of shots
    ``eps01`` : 0 state detection infidelity, aka set0 meas1
    ``eps10`` : 1 state detection infidelity, aka set1 meas0

    Returns
    -------
    dict with keys walls_tot, walls_tot_err, walls_tot_mit, walls_tot_mit_err : 
        walls_tot: unmitigated domain wall number
        walls_tot_err: unmitigated domain wall number error
        walls_tot_mit: mitigated domain wall number
        walls_tot_mit_err: mitigated domain wall number error
    """

    #bringing the strings to integers - this can be done better for sure.
    bitstring_data=np.array([np.int_([*np.array(bitpattL)[i]]) for i in range(len(bitpattL))])

    # bitstring_data`` : array of bits
    #    Actual bitstring data - the first dimension is the number of samples, the second system size.

    Cf=conf_matrix_inverse2x2(eps01, eps10)
    L=bitstring_data.shape[1]
    walls=np.zeros((L,))
    walls_mit=np.zeros((L,))
    craw_count=0
    for i in range(L):
        tmp=bitstring_data[:,[i,(i+1)%L]] #concentrating on two sites
        tmp1=tmp[:,0]*2+tmp[:,1] #binary to decimal
        praw=np.array([sum(tmp1==i) for i in range(4)]) #two site counts
        pmit=Cf@praw #mittigated two site counts
        craw=praw[0]+praw[3] #domain wall number -> 00 and 11 configurations contrubute
        cmit=pmit[0]+pmit[3] #mitigated domain wall number
        craw_count=craw_count+craw
        walls[i], walls_mit[i] =craw/sum(praw), cmit/sum(praw) #taking the average
    walls_tot, walls_tot_err, walls_tot_mit, walls_tot_mit_err=sum(walls), sum(walls)/np.sqrt(craw_count), sum(walls_mit), sum(walls_mit)/np.sqrt(craw_count) #average over the whole system and errors that propagate

    #return  {"walls_tot":walls_tot, "walls_tot_err":walls_tot_err,   "walls_tot_mit":walls_tot_mit, "walls_tot_mit_err":walls_tot_mit_err}

    return {'measured': [ walls_tot,walls_tot_err ], 'mitigated': [walls_tot_mit, walls_tot_mit_err] }
    
    
 
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    np.set_printoptions(precision=4)           
    inpF=args.expName
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
         
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop3

    nAtom=expMD['payload']['num_atom']
    momentV=expD["domain_wall_dens_moments"]
    mySpam=expMD['zne_wall_dens']['readout_error']
    shots=expMD['submit']['num_shots']
    print('\nM:assumed SPAM probability  s0m1: %.2f   s1m0: %.2f'%(mySpam[0],mySpam[1]))
    print('total shots %d'%shots)
    #....  unpack 'real' measurements
    print('\nM:dump data for ideal+SPAM')
    countsV=expD['ranked_counts']
    hexpattV=expD['ranked_hexpatt']

            
    bitpattL=flatten_ranked_hexpatt_array(countsV,hexpattV,nAtom)        
    if args.verb>1 : print('dump bitpattL:', len(bitpattL), bitpattL[:3])

    for i in range(2):       
        if i==0:  print('M: ground truth')
        if i==1:  print('M: SPAM included')
        X=momentV[i]
        print('     mean=%.2f +/-%.2f, std=%.2f  +/-%.2f'%(X[0],X[1],X[2],X[3]))
        
    
    print('\nM:now apply SPAM correction...')
    outD=wall_count_mitig_v1(bitpattL,eps01=mySpam[0],eps10=mySpam[1])
    # append truth
    print('\n dataset',inpF)
    print('\n SPAM mitigation by Milo',inpF)
    outD['ideal']=momentV[0,:2]
    outD['jan_meas']=momentV[1,:2]
    pprint(outD)

    print('\nrelative errors:')
    for x in sorted(outD):
        val,err=outD[x]
        print('%s  nSig=%.1f'%(x, val/err))

    
 
