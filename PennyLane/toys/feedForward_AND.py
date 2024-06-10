#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
feed-forward boolen AND of 2 measurements

Summary:
 y=w0*w1
 w0,w1  in [0,1]

Note, works with probabilities, NOT on expval()

0: ──RY(1.54)──┤↗├─────────┤       
1: ──RY(1.96)───║───┤↗├────┤       
2: ─────────────║────║───X─┤  Probs
                ╚════║═══╣         
                     ╚═══╝         
'''

import pennylane as qml
from time import time, sleep

import numpy as np
import os

import argparse

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-n','--num_shot',type=int,default=2000, help="shots, aka sequence length")
    parser.add_argument('-w','--weights', default=[0.9, 0.7], nargs=2, type=float, help='hardcoded weights in [0,1] ')

    
    parser.add_argument( "-r","--randomData",   action='store_true', default=False, help="(optional)replace all inputs by random values")

    args = parser.parse_args()
    if args.randomData:
        print('Random inputs')
        eps=0.005
        args.weights=np.random.uniform(eps, 1.-eps, size=len(args.weights))
       
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args


#...!...!....................
def build_circ_GGAE_2q(args,dev):
    wV=np.array(args.weights)
    assert np.all(wV <= 1), "Not all elements are smaller than 1."
    assert np.all(wV >=0)
 
    th0,th1=np.arccos(1-2*wV)
    @qml.qnode(dev)
    def circuit():
           
        qml.RY(th0,0)
        qml.RY(th1,1)
                
        m0 = qml.measure(0)
        m1=qml.measure(1)
    
        #.... bit processing
        and01=(m1 == 1) * ( m0==1)
        qml.cond(and01==1, qml.X)(2)
        
        return qml.probs(wires=2)
    return circuit


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=4)
    args=commandline_parser()
    shots=args.num_shot
    dev = qml.device("default.qubit",wires=3,shots=shots)

    circ=build_circ_GGAE_2q(args,dev)

    print(qml.draw(circ)())

    print('job started, dev:',dev)
    T0=time()
    mp=circ()[1]    
    elaT=time()-T0
   
    print('P:  ended elaT=%.1f sec\n '%(elaT))
  
    sErr=1/np.sqrt(shots)
    print('M: done, shots=%d  mp=%.3f  1/sqrt(shots)=%.3f '%(shots,mp,sErr))

    w0,w1=args.weights
    ty=w0*w1

    diff=ty-mp
    if  abs(diff)<0.03 : okStr='--- PASS ---'
    elif  abs(diff)<0.1  : okStr='.... poor ....' 
    else: okStr='*** FAILED ***'
  
    print('inp: xV= %3f, %3f; out:   my=%.3f ty=%.3f  %s  diff=%.3f'%(w0,w1,mp,ty,okStr,diff))
