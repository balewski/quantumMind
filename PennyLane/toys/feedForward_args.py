#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Use PennyLane
Hardcoded circuit, w/o parameterization

Scaling of input : 'S'

Summary:
Gen(x) -->Scale(d0,d1) -->Eval

x in [-1,1], wV=[d0,d1] in [0,1]
y= A*x+B  where A=(1-d0-d1),  B=d1-d0

'''

from time import time, sleep

import numpy as np
import os
import pennylane as qml

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('-n','--num_shot',type=int,default=2000, help="shots, aka sequence length")
    parser.add_argument('-x','--inputs', default=[-0.3], nargs=1, type=float, help=' input values in [-1,1]')
    parser.add_argument('-w','--weights', default=[0.9, 0.7], nargs=2, type=float, help='hardcoded weights in [0,1] ')

    parser.add_argument( "-r","--randomData",   action='store_true', default=False, help="(optional)replace all inputs by random values")

    args = parser.parse_args()
    if args.randomData:
        print('Random inputs')
        eps=0.005
        args.inputs=np.random.uniform(-1.+eps, 1.-eps, size=len(args.inputs))
        args.weights=np.random.uniform(eps, 1.-eps, size=len(args.weights))

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))   
    return args


#...!...!....................
def inpdata_to_angles(xV):
    if not  isinstance(xV, np.ndarray):
        xV=np.array(xV)

    assert np.all(xV <= 1), "Not all elements are smaller than 1."
    assert np.all(xV >= -1)
    thetaV=np.arccos(xV)
    print('inpData: xV:',xV, ' thV:',thetaV)
    return thetaV

#...!...!....................
def relweights_to_angles(wV):
    if not  isinstance(wV, np.ndarray):
         wV=np.array(wV)
  
    #print('QW2A: inp:',wV,type(wV))
    assert np.all(wV <= 1), "Not all elements are smaller than 1."
    assert np.all(wV >=0)
    alpha=np.arccos(1.- 2*wV)
    print('inpRelWeighs:',wV,' alpha:',alpha)
    return alpha

    print('inpData: xV:',xV, ' thV:',thetaV)
    return thetaV



#...!...!....................
def build_circ_GSE_1q(args,dev):
    th=inpdata_to_angles(args.inputs)[0]
    alV=relweights_to_angles(args.weights)

    @qml.qnode(dev)
    def circuit():
        qi=0
        #....  encode input
        qml.RY(th,qi)
        #...   scale  y=A*x+B
        m = qml.measure(qi) 
        qml.cond(m == 0, qml.RY)(alV[0],qi)
        qml.cond(m == 1, qml.RY)(alV[1],qi) 
        return qml.expval(qml.PauliZ(qi))
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
    dev = qml.device("default.qubit",wires=1,shots=shots)

    circ=build_circ_GSE_1q(args,dev)

    print('circ()=%.3f'%circ())
    print(qml.draw(circ)())
   
    
    print('job started, dev:',dev)
    T0=time()
    my=circ()    
    elaT=time()-T0
   
    print('P:  ended elaT=%.1f sec\n '%(elaT))
  
    sErr=1/np.sqrt(shots)
    print('M: done, shots=%d  mEV=%.3f  1/sqrt(shots)=%.3f '%(shots,my,sErr))
   
    x=args.inputs[0]
    dV=args.weights
    pin=(1-x)/2
    ty=x*(1-dV[0]-dV[1]) + dV[1]-dV[0]

    diff=ty-my
    if  abs(diff)<0.03 : okStr='--- PASS ---'
    elif  abs(diff)<0.1  : okStr='.... poor ....' 
    else: okStr='*** FAILED ***'
  
    print('inp: x=%.3f, pin=%.3f ; out:  my=%.3f  ty=%.3f %s diff=%.3f'%(x,pin,my,ty,okStr,diff))
    
    
