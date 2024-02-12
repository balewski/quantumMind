__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from decimal import Decimal
from collections import Counter
from bitstring import BitArray
from pprint import pprint

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField

from toolbox.Util_stats import do_yield_stats
from toolbox.UAwsQuEra_job import  build_ranked_hexpatt_array

#...!...!..................
def compute_nominal_rb(Omega, Delta):  # both  inputs in rad/sec , SI
    # Compute the Rydberg blockade radius , see https://arxiv.org/pdf/1808.10816.pdf 
    C6=5.42E-24
    return (C6 / np.sqrt((Omega)**2 + Delta**2))**(1/6)  # in m, SI
''' test solution: 
Omega=2.5 * 2*np.pi *1e6
['omega=%.2e'%Omega,'Rb=%.2e'%compute_nominal_rb(Omega,0), ]
['omega=1.57e+07', 'Rb=7.46e-06']
'''

'''
QuEra, Jonathan Wurtz:
I think that having no factor of two is more correct. The operator Delta*n has eigenvalues 0 and Delta, with a spectral difference of Delta. Alternatively, the Rabi term, with its factor of 2 included, has eigenvalues +/- Omega/2, with a spectral difference of Omega. Thus, the energy scale for Omega should also be Omega and not 2*Omega.
'''
#...!...!..................
def register_to_numpy(register):
    xL=register.coordinate_list(0)
    yL=register.coordinate_list(1)
    posL=[]
    for x,y in zip(xL,yL):
        #print('xy',x,y)            
        posL.append([float(x),float(y)])
    return np.array(posL)  # units (m), SI

#...!...!..................
def register_from_numpy(atoms_xy):
    atoms = AtomArrangement()
    for x,y in atoms_xy:
        #print('xy',x,y)            
        atoms.add([Decimal(x),Decimal(y)])
            

#...!...!..................
def drive_to_numpy(drive):
    #print('U:dump drive filed Amplitude(time) :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())
    times=[float(x) for x in drive.amplitude.time_series.times()]
    amplitudes=[float(x) for x in drive.amplitude.time_series.values()]
    detunings=[float(x) for x in drive.detuning.time_series.values()]
    phases=[float(x) for x in drive.phase.time_series.values()]
    #print('my ampl:',amplitudes)
    #print('my times:',times)
    #print('my detune:',detunings)
    dd=np.array([times,amplitudes,detunings,phases])
    return dd

#...!...!..................
def drive_from_numpy(fields):
    times,amplitudes,detunings,phases=fields
    return  DrivingField.from_lists(times,amplitudes,detunings,phases)


"""Stitches two driving fields based on TimeSeries.stitch method.
The time points of the second DrivingField are shifted such that the first time point of
the second DrifingField coincides with the last time point of the first DrivingField.
The boundary point value is handled according to StitchBoundaryCondition argument value.

Use case:
  Ha=ramp_drive_waveform(t_up,omega_max,delta_begin,'pre')
  Hb=mid_drive(t_vary,omega_max,delta_begin,delta_end,pd['detune_shape'] )
  Hc=ramp_drive_waveform(t_down,omega_max,delta_end,'post')
  H=Ha.stitch(Hb).stitch(Hc)

"""
#...!...!..................
def ramp_drive_waveform(t,omega,delta,rdir,phi=0.0): # initial waveform 
    Omegas = TimeSeries() ;  Deltas = TimeSeries() ;   Phis = TimeSeries()
    if rdir=='pre':
        Omegas.put(0.0, 0.0).put(t,omega)
    if rdir=='post':
        Omegas.put(0.0, omega).put(t,0.0)
    Deltas.put(0.0, delta).put(t,delta)
    Phis.put(0.0, 0.0).put(t,phi)
    H = DrivingField(amplitude=Omegas, phase=Phis, detuning=Deltas)
    return H

#...!...!..................
def mid_drive_waveform(t_tot,omega,delta_begin,delta_end,rdeltaL,phi=0.0):
    Omegas = TimeSeries() ;  Deltas = TimeSeries() ;   Phis = TimeSeries()
    # append 0-th & last to steps and set detune to min & max, respectively
    #print('tt',rdeltaL)
    rdeltaL=[0.0]+rdeltaL+[1.0]
    nStep=len(rdeltaL)
    t_step=t_tot/(nStep-1) 
    delta_diff= delta_end - delta_begin
    for i in range(nStep):
        ti=t_step*i
        di=delta_begin+ delta_diff*rdeltaL[i]
        #print('ddt',i,ti,di)
        Omegas.put(ti,omega)
        Deltas.put(ti,di)
        Phis.put(ti,phi)
    H = DrivingField(amplitude=Omegas, phase=Phis, detuning=Deltas)
    return H

#...!...!..................
def scar_drive_waveform(t1,t_flat,omega,delta,phi=0.0): # initial waveform 
    Omegas = TimeSeries() ;  Deltas = TimeSeries() ;   Phis = TimeSeries()
    t2=t1+t_flat
    t3=t2+t1
    Omegas.put(0.0, 0).put(t1,omega).put(t2,omega).put(t3,0)
    Deltas.put(0.0, delta).put(t1,0).put(t2,0).put(t3,0)
    Phis.put(0.0, 0.0).put(t1,phi).put(t2,phi).put(t3,phi)
    H = DrivingField(amplitude=Omegas, phase=Phis, detuning=Deltas)
    return H

#...!...!..................
def raw_experiment_postproc(meta,expD,rawCounter,verb):             
    pd=meta['payload']
    amd=meta['postproc']
    nClust=pd['num_clust']
    nAtom= pd['num_atom_in_clust']
    totAtom=pd['num_atom']
    assert nClust*nAtom==totAtom

    ''' break bit-strings into clusters
        QA:  count broken strings with e and throw them away
    '''
    solCnt={}
    failCnt={}
    nBig=len(rawCounter)

    uShots=0  # used shots
    fShots=0
    for bigS,mshot in rawCounter.items():
        assert len(bigS)==totAtom
        measL=[ bigS[i:i+nAtom] for i in range(0, totAtom,nAtom) ]
        #print('measL:',measL)
        for measS in measL:
            if 'e' in measS :  
                if measS not in failCnt: failCnt[measS]=0
                failCnt[measS]+=mshot
                fShots+=mshot
                continue
            uShots+=mshot
            if measS not in solCnt: solCnt[measS]=0
            solCnt[measS]+=mshot
        #print('aa',bigS,mshot,'nSol=',len(solCnt),len(failCnt),uShots)

    nSol=len(solCnt)
    nFail=len(failCnt)
    print('nAny=%d, nSol=%d  nFail=%d, shots used=%d failed=%d '%(nBig,nSol,nFail,uShots,fShots))
    #print('all sol:');pprint(solCnt)
    print('all %d e-fail:'%nFail);
    if nFail>0:
        j=0
        for key, value in failCnt.items():
            print(j,key,value)
            j += 1
            if j>5: break

    solCounter=Counter(solCnt)

    #...... collect meta-data
    amd['num_atom']=nAtom
    amd['num_sol']=nSol
    amd['num_fail']=nFail
    amd['used_shots']=uShots

    #... correct other MD
    if 'clust' not in pd['info']:
        txt=pd['info'].split(",")
        txt[0]='atoms:%d'%nAtom
        txt.append('clust:%d'%nClust)
        pd['info']=','.join(txt)        
        #print('txt=',txt); aa

    '''  repack  'rgggrgr' --> numpy
    store them orderd by number of shots
    additional record keeps node index (aka MIS rg-string converted to 1 int)
    next is sparse packing
    '''
    dataY,pattV,hammV,nAtom2=build_ranked_hexpatt_array(solCounter)
    assert nAtom==nAtom2
    

    ''' DISCARD 
    rankedSol=solCounter.most_common(nSol)
    NB=nSol # number of bitstrings,  this is sparse encoding 
    dataY=np.zeros(NB,dtype=np.int32)  # measured shots per pattern
    pattV=np.empty((NB), dtype='object')  # meas bitstrings as hex-strings (hexpatt)
    hammV=np.zeros(NB,dtype=np.int32)  # hamming weight of a pattern

    hexlen=int(np.ceil(nAtom / 4) * 4)
    pre0=hexlen-nAtom
    print('RYA: dataY:',dataY.shape, 'hexlen=%d, pre0=%d'%(hexlen,pre0))
    patt0='0'*pre0 # now pattern is divisible by 4
    for i in range(nSol):
        rgpatt,mshot=rankedSol[i]
        patt=rgpatt.replace('g', '0').replace('r', '1')
        assert 'e' not in patt , "can't handle missing atoms at this stage"
        A=BitArray(bin=patt0+patt)
        hw=A.bin.count('1')  # Hamming weight
        if i<10: print('ii',i,rgpatt,patt,'hex=%s hw=%d mshot=%d'%(A.hex,hw,mshot))
        dataY[i]=mshot
        pattV[i]=A.hex
        hammV[i]=hw
    '''
    
    if verb>1:
        print('RYA: dump dataY:',dataY)
        print('pattV',pattV)
        print('hammV',hammV)

    
    expD['ranked_counts']=dataY   # [NB]
    expD['ranked_hexpatt']=pattV  # [NB]
    expD['ranked_hammw']=hammV    # [NB]

    dataP=do_yield_stats(dataY.reshape(1,-1)).reshape(3,-1)
    if verb>1:
        print('dataP:',dataP)
    expD['ranked_probs']=dataP   # [PEY,NB]


        
#...!...!..................
def states_energy(task):             
    pd=task.meta['payload']
    amd=task.meta['postproc']
    nAtom= pd['num_atom_in_clust']
    nSol=amd['num_sol']
    fact=9e18  # spead of light squared
    hbar=1.05457182e-34 # in (J s)
    C6=5.42E-24
    to_eV=1.602176634e-19 # 1eV in J

    tV,ampls,detuns,phases=task.getDrive()
    detuneLast=detuns[-1]
    #print('SE:last detune:',detuneLast)
    
    hexpattV=task.expD['ranked_hexpatt']
    pos=task.expD['atoms_xy'][:nAtom]
    #print('SE: pos:',pos)
    assert nSol==hexpattV.shape[0]

    energyV=np.zeros(nSol)
    #.. loop over solutions
    for k in range(nSol):
        hexpatt=hexpattV[k]
        A=BitArray(hex=hexpatt)[-nAtom:]  # clip leading 0s
        # ...  count 1s
        hw=A.bin.count('1')  # Hamming weight
        eneRydb=detuneLast*hw
        # ... select pairs
        L=[  i  for i, bit in enumerate(A) if  bit]
        pairs = []        
        for i in range(hw):
            for j in range(i+1, hw):
                pairs.append((L[i], L[j]))
        enePairs=0.
        #print('patt:',A.bin, '\pairs:',pairs)
        for i,j in pairs:
            delPos=pos[i]-pos[j]
            r2=np.sum(delPos**2)
            delE=C6/r2**3
            #print('i,j',i,j,pos[i],pos[j],delE)
            enePairs+=delE
        #print(k,'ene Rydb, Pairs:',eneRydb, enePairs ,hw,len(pairs),'rydb/Pairs:',eneRydb/enePairs)
        energyV[k]= (-eneRydb+enePairs) * hbar /to_eV
    task.expD['ranked_energy_eV']=energyV
    amd['energy_range_eV']=[ np.min(energyV),np.max(energyV)]
    print('SE:done computing energy for %d states, energy range:'%(nSol),amd['energy_range_eV'])

'''  (h-bar), also known as the reduced Planck constant, is approximately 1.05457182 x 10^-34 joule-seconds (J·s) in the International System of Units (SI). 

E.g. the Van der Waals energy of 2 atoms spaced by 6 um would be now
5.24e-24/(6e-6)^6*1.05e-34*1e12=1.18e-14 (pJ) (pico-Joule)

'''
        
    
