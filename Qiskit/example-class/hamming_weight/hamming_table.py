#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
computes lookup table for Hamming distance 

'''
import time,os,sys
from pprint import pprint

import numpy as np
import qiskit as qk
from bitstring import BitArray

        
#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    n=3  # number of bits in the word
    m=2 # bits for hamming weight
    mxval=1<<n
    acc={'b0':'','b1':''}
    print('Hamming table for %d bits has size of %d'%(n,mxval))
    print('i  bin --> ham dec ')
    print('   abc     de')
    for i in range(mxval):
        A=BitArray(uint=i, length=n)        
        hamm=(A).count(1) # Hamming weight
        H=BitArray(uint=hamm, length=m)        
        print('%d  %s     %s  %d'%(i,A.bin,H.bin,hamm))
        acc['b0']+='%d'%H[-1]
        acc['b1']+=H[-2:-1].bin

    print('tables',acc)
    print('M:end')


