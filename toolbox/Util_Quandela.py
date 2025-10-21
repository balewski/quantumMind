#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import perceval as pcvl
from time import time, sleep
from tqdm.notebook import tqdm
from pprint import pprint
import numpy as np

def generate_binary_list(nb):
    return ['0' * nb, '1' * nb] + [format(i, f'0{nb}b') for i in range(1, 2**nb - 1)]

#...!...!.................... 
def decode_sampler_result(resD,num_qubit,verb=1):
    nLab= 1+ (1<<num_qubit)
    bitStrL=generate_binary_list(num_qubit)+['bad']
    outCnt=np.zeros(nLab,dtype=int)    
    print(resD)
    
    perf=None  # local sim and cloud sim have different dict
    for xx in ['physical_perf', 'global_perf']:
        if xx in resD:
            perf=resD[xx]
            break
    assert perf!=None

    fockR=resD['results' ]# number of photons in each mode.    
    for foStt, count in fockR.items():
        #print("photon fock state:", foStt, "Count:", count)        
        bitStr=fockState_to_bitStr(foStt)
        if verb: print(foStt,bitStr,":",count)
        i= bitStrL.index(bitStr)
        outCnt[i]+=count

    return outCnt,perf,bitStrL

#...!...!.................... 
def fockState_to_bitStr(basic_state):
    """
    Convert a Perceval BasicState to a qubit state as a bitstring,
    assuming dual-rail encoding.
    """
    # Ensure even number of modes (dual-rail pairs)
    if len(basic_state) % 2 != 0:
        raise ValueError("BasicState length must be even for dual-rail encoding.")

    # Convert to qubit state as a bitstring
    qubit_state = ''
    for i in range(0, len(basic_state), 2):
        # Each qubit is represented by a pair of modes
        if basic_state[i] == 1 and basic_state[i+1] == 0:
            qubit_state += '0'  # Photon in first mode → |0>
        elif basic_state[i] == 0 and basic_state[i+1] == 1:
            qubit_state += '1'  # Photon in second mode → |1>
        else:
            return 'bad'
    
    return qubit_state

#...!...!....................
def bitStr_to_dualRailState(bits):
    """
    Convert a qubit state bitstring to a Perceval BasicState,
    assuming dual-rail encoding.
    """
    # Check input validity
    if not all(b in '01' for b in bits):
        raise ValueError("Input must be a bitstring containing only '0' and '1'.")

    # Convert bitstring to dual-rail BasicState
    state = []
    for b in bits:
        if b == '0':
            state.extend([1, 0])  # |0> → |1, 0>
        elif b == '1':
            state.extend([0, 1])  # |1> → |0, 1>
    
    return pcvl.BasicState(state)




#...!...!.................... 
def monitor_async_job(remote_job, numSec=10):
    T0=time()
    i=0
    while True:
        jstat=remote_job.status()
        elaT=time()-T0
        print('M:i=%d  status=%s, elaT=%.1f sec'%(i,jstat,elaT))
        if jstat=='SUCCESS': break
        if jstat=='ERROR': exit(99)
        i+=1; sleep(numSec)
    return

#...!...!.................... 


#...!...!.................... 

#...!...!.................... 


#=================================
#   U N I T   T E S T
#=================================

if __name__=="__main__":
    print('testing Perceval utility functions')
    print('ver:',pcvl.__version__)
    
    # Example usage
    fockStt = pcvl.BasicState([1, 0, 0, 1])  # Should represent qubit state |0⟩ |1⟩
    bitStr=fockState_to_bitStr(fockStt)
    
    print("\nINPUT BasicState:", fockStt)
    print("Qubit State as Bitstring:", bitStr)

    fockStt = pcvl.BasicState([1, 2, 0, 1])  # Should represent qubit state |0⟩ |1⟩
    bitStr=fockState_to_bitStr(fockStt)
    
    print("\nINPUT BasicState:", fockStt, bitStr)

    # Example usage
    bitStr = '0101'  # Should represent BasicState |1,0,0,1,1,0,0,1>
    fockStt=bitStr_to_dualRailState(bitStr)

    print("\nINPUT Bitstring:", bitStr)
    print("BasicState:", fockStt)

    
    
