#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
given initial bitstring of length N and weight W
* generate offsprings with W-1
* for each offspring produce all ancestors
* find most probable ancestor
* compare it to initial string

* select ISs
'''

from bitstring import BitArray
import numpy as np
from pprint import pprint

#...!...!....................
def gen_offsprings(A):
    lA=len(A)
    wA=A.count(True) # Hamming weight
    print('  inp',A.bin,'W=%d L=%d'%(wA,lA))
    #... indices of all 1-bits
    iL = [i for i, bit in enumerate(A) if bit] 
    print('1-idx:',iL)

    childL=[i for i in range(wA) ] # prime storage
    k=0
    for i, bit in enumerate(A):
        if not bit: continue        
        C=A.copy()
        C[i]=0
        print(i,type(bit),A[i],bit,'C=',C.bin)
        childL[k]=C
        k+=1
    return childL
        


#...!...!....................
def reco_mpv_ancestor(childL):
    tmpD={}
    for A in childL:
        print('\naa',A.bin)
        # try to replace every 0 -->1
        for i, bit in enumerate(A):
            if bit : continue
            C=A.copy()
            C[i]=1
            k=C.uint
            #print(i,A,bit,'C=',C.bin,C.bin==trgstr,'k=',k)
            
            if k not in tmpD: tmpD[k]=0
            tmpD[k]+=1
        #break
    print('tmpD',tmpD)
    max_key = max(tmpD, key=tmpD.get)
    A=BitArray(uint=max_key,length=A.length)
    return A.bin, tmpD[max_key]

#...!...!....................
def select_IndepSet(bitstrL):
    outL=[]
    for bitstr in bitstrL:
        if '11' in bitstr.bin: continue
        outL.append(bitstr)
    print('found %d valid ISs:'%len(outL),outL)
        
#...!...!....................


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    trgstr='0101100'
    
    A=BitArray(bin=trgstr)
    misID=A.uint
    wA=A.count(True) # Hamming weight
    print('M: trg',trgstr,misID,wA)
    childL=gen_offsprings(A)
    print('M: childL:',childL)

    recstr,nOccur=reco_mpv_ancestor(childL)
    print('reco str=%s nOccur=%d'%(recstr,nOccur),recstr==trgstr)


    select_IndepSet(childL)
