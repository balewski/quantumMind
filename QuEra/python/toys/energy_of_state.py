#!/usr/bin/env python3
from itertools import combinations
import numpy as np
# see https://queracomputing.github.io/Bloqade.jl/dev/hamiltonians/ for units.

# Aquila HW  units for energy are MHz
C6_aquila=862690*2*np.pi # in (MHz um^6) for |r>=|70S_1/2> of 87^Rb

# example experiment
bitstrL=['100', '011']  # measured states
atomPos=np.array([(0,0),(100,0),(104,0)]) # (um)
detuneLast=[ 1320, 0, 0 ]  # (rad/us )

for key in bitstrL:
    idx1 = [i for i, c in enumerate(key) if c == '1']
    # Get all unique pairs of indices
    idx12 = list(combinations(idx1, 2))
    #print(key,idx1,idx12)
    
    # ...  rydberg energy  , blockade energy
    eneRydb=0.
    for i in idx1: eneRydb-=detuneLast[i]
    
    eneBlock=0.
    for i1,i2 in idx12:
        delPos=atomPos[i1] -atomPos[i2]
        r2=np.sum(delPos**2) 
        ene12=C6_aquila/r2**3   
        eneBlock+=ene12
    print('key=%s  eneRydb=%.1f (MHz)  eneBlock=%.1f  (MHz)'%(key,eneRydb,eneBlock))
