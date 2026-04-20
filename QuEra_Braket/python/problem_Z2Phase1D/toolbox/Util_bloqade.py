from collections import OrderedDict
from itertools import combinations
import numpy as np

#...!...!..................

def invert_keys(D):  # maps g-->0, r -->1  to stick with computer science convention
    # Lambda function to invert the '0's and '1's
    invert_binary = lambda s: s.replace('0', '2').replace('1', '0').replace('2', '1')
    D_inverted = [OrderedDict((invert_binary(k), v) for k, v in d.items()) for d in D]
    return D_inverted

#...!...!..................
def append_state_energy(counts,atomPos_um,detune_rad2us, verb=0): # modiffies counts in-place
    # Aquila HW  units for energy are MHz
    C6_aquila=862690*2*np.pi # in (MHz um^6) for |r>=|70S_1/2> of 87^Rb

    # ..... sanity
    nAtom=len(detune_rad2us)
    assert atomPos_um.ndim==2
    assert atomPos_um.shape[0]==nAtom
    assert len (next(iter(counts))) ==nAtom #  bitstring len
    minEne=1e99; bestKey=None
    # ..... compute energy
    for key in counts:
        hw=key.count('1')
        idx1 = [i for i, c in enumerate(key) if c == '1']
        # Get all unique pairs of indices
        idx12 = list(combinations(idx1, 2))
        #print(key,idx1,idx12)
        
        eneRydb=0.  #  rydberg energy  
        for i in idx1: eneRydb-=detune_rad2us[i]
        
        eneBlock=0.  # blockade energy
        for i1,i2 in idx12:
            delPos=atomPos_um[i1] -atomPos_um[i2]
            r2=np.sum(delPos**2)
            ene12=C6_aquila/r2**3   
            eneBlock+=ene12
        eneTot=eneRydb+eneBlock
        if minEne>eneTot:
             minEne=eneTot; bestKey=key; ctag='*'
        else: ctag=' '
        if verb>0:  print('Hamil: state=%s%s  energy(MHz): total=%7.1f    Rydb=%7.1f   Block=%7.1f   HW=%d'%(key,ctag,eneTot,eneRydb,eneBlock,hw))
        #.... (optional) add energy to OrderedDict
        if isinstance(counts, OrderedDict):
            counts[key]= (counts[key], eneTot,eneRydb,eneBlock, hw)
        else:
            assert verb >0 # no output
    if verb>0:
        print('min_energy: key=%s  energy(MHz)=%.1f'%(bestKey,minEne)) 
        print()
        
''' testing add_state_energy(.)

if 1:  # white-paper dist=4um : matching rydberg=blockade energy
    bitstrL=['110', '001','111']  # measured states
    atomPos=np.array([(0,0),(4,0),(150,0)]) # (um)
    detuneLast=[ 0,0, 1320 ]  # (rad/us )

if 0:  # white-paper dist=16 um
    bitstrL=['110', '001']  # measured states
    atomPos=np.array([(0,0),(16,0),(150,0)]) # (um)
    detuneLast=[ 0,0, 0.32 ]  # (rad/us )    
add_state_energy(bitstrL,atomPos,detuneLast)
'''
