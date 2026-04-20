#!/usr/bin/env python3
# coding: utf-8

'''
6-spin QUBO is subset of Ana's problem.
QUBO definition

E = 3*n0 + 6*n1  + 10*n2 +20*n3 + 8*n0n2 + 20*n3 + 4 * n4 + 8 * n5
      + 200*Exor(n0,n1)  +  200*Exor(n2,n3)  +  200*Exor(n4,n5) 

     where Exor(a,b)= 2a*b -a -b   is blocking constraint

'''
# use global detune to balance blockade
from bloqade import start
import numpy as np
import os, time
from bitstring import BitArray

from toolbox.Util_bloqade import invert_keys, append_state_energy

#...!...!....................
def postproc_to_logicalIds(counts):
    nAtom=len(atomLogId)
    outL=[]
    for key, val in counts.items():
        A=BitArray(bin=key)
        #print(key,val,'A.uint=',A.uint)
        lsbf=['-' for i in range(nAtom)]
        for i,j in enumerate(atomLogId):
            lsbf[j]=int( A[i] )
        #print('lsbf:',lsbf)
        outL.append( (lsbf,val))
    return outL

#...!...!....................
def print_tiger3( L):
    print('     t0  t1  t2')
    print('CN0:  %d   %d  %d'%(L[0],L[2],L[4]))
    print('CN1:  %d   %d  %d'%(L[1],L[3],L[5]))
    print('sequence: ',end='')
    for i in range(6):
        if L[i]: print('n%d '%i,end='')
    print()

#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    shots=1000
    # Define relevant parameters for the lattice geometry and pulse schedule
    atomLogId=[4,5,1,0,2,3]
    atomPos=[ (0,0), (4.9,0),(10.6,0), (15.4,0), (23.8,0), (28.7,0) ]
    localScale=[ 0.941, 0.706, 0.824, 1.0, 0.588, 0.0]  
    rabi_amplitude = [0.0, 15., 15., 0.0]
    delta_glob = [-30, -30, 160, 160]
    delta_local = [ 0, 0, 34, 34 ]
    au2MHz=2.  # for printout only
    durations = ["ramp_time", "sweep_time", "ramp_time"]

    atomPos=np.array(atomPos)
    nAtom = atomPos.shape[0]
    register = start.add_position(atomPos)
    print('register size:',register.n_sites,register)
    print('logical IDs:',atomLogId)

    atomHardId=[i for i in range(nAtom) ]
    detuneLast=['-' for i in range(nAtom) ]

    print('local detune scale:',localScale)
    print('Combinded\n hardId, logId, detune start, detune end')
    for k in range(nAtom):
        dL=delta_glob[0] + delta_local[0]*localScale[k]
        dR=delta_glob[-1] + delta_local[-1]*localScale[k]
        print('%d   %d  %.1f  %.1f'%(k,atomLogId[k],dL,dR))
        detuneLast[k]=dR

    prog = (
        register
        .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, rabi_amplitude)
        .detuning.uniform.piecewise_linear(durations, delta_glob)
        .location(atomHardId,localScale).piecewise_linear(durations, values=delta_local)
    )

    any_bitstrings = [format(i, '0{}b'.format(nAtom)) for i in range(2 ** nAtom)]
    append_state_energy(any_bitstrings,atomPos,detuneLast,verb=1)

    #...... vary duration of the sweep time
    for tsweep in [3,6,10,30] :
        job = prog.assign(ramp_time=0.5,sweep_time=tsweep)
        t0=time.time(); print('\nrun nAtom=%d  shots=%d  tsweep=%.1f us ...'%(nAtom,shots,tsweep),end='')
        emu = job.bloqade.python().run(shots)
        print('  done run elaT=%.1f sec'%(time.time()-t0))
        report = emu.report()

        counts=invert_keys(report.counts)[0]   #  is '1'=rydberg
        #print('Counts:',counts)

        append_state_energy(counts,atomPos,detuneLast,verb=0)
        for key,val in counts.items():        
            m,e,_,_,hw=val
            print('raw key=%s  mshot=%3d  ene=%7.1f (MHz)  HW=%d'%(key,m,e,hw))

        #continue
        outL=postproc_to_logicalIds(counts)
        #print('\nM:',outL)

        for j, (logIdL, valT) in enumerate(outL):
            #print('aaa',logIdL, valT)
            mshot,ene=valT[:2]
            if mshot< shots/20: break
            print('\nsolution: %d , mshot: %d  ene: %7.1f (a.u.)'%(j,mshot,ene/au2MHz))
            print_tiger3( logIdL)
  
        #break # tmp
    print('M:done')

