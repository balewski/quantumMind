#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Test Feed-forward
should produce 000 or 111 states from bell-state and 1 conditional-x gate


'''
import perceval as pcvl
from perceval.algorithm import Sampler
cnotH = pcvl.catalog["heralded cnot"].build_processor()
import numpy as np
from time import time,sleep

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    num_qubit=4
    nMode=2*num_qubit
    minPhoton=num_qubit  
    isIdeal= True
    
    if isIdeal:
        shots=50_000 
        proc = pcvl.Processor("SLOS",nMode)
    else:
        shots=50_000_000_000  
        #proc = pcvl.RemoteProcessor("sim:ascella",m=nMode)
        proc = pcvl.RemoteProcessor("sim:belenos",m=nMode)

    # Set up the circuit: H gate + CNOT to create Bell state
    proc.add(0, pcvl.BS.H())  # Hadamard on q0
    #1proc.add(0, pcvl.PERM([1, 0]))  # 1 state on q0
    proc.add(0, cnotH)        # CNOT with heralding


    if 1:  # Feed-forward X correction on q3  based on measurement of Alice's qubit 1
        nModes=2; # the size of the circuit that will be applied conditionally.
        modeOff=2; # The first mode index (in the global processor) where this feed‑forward block will be inserted.
        ff_X = pcvl.FFCircuitProvider(nModes, modeOff, pcvl.Circuit(2))
        #                     ( measurement_pattern, conditional_circuit)
        ff_X.add_configuration([0, 1], pcvl.PERM([1, 0]))
        proc.add(2, pcvl.Detector.pnr())
        proc.add(3, pcvl.Detector.pnr())
        proc.add(2, ff_X)
    

    if 0 :
        # Feed-forward Z correction on q2  based on measurement of Alice's qubit 0
        modeOff=3  # this selects q3, since it is 2*nModes+modeOff from the top of processor
        phi = pcvl.P("phi")
        ff_Z = pcvl.FFConfigurator(nModes, modeOff, pcvl.PS(phi), {"phi": 0})
        ff_Z.add_configuration([0, 1], {"phi": np.pi})
        proc.add(0, pcvl.Detector.pnr())
        proc.add(1, pcvl.Detector.pnr())
        proc.add(0, ff_Z)
    
   
    # X gate on qubit 2 (dual-rail modes 4 and 5)
    #proc.add(4, pcvl.PERM([1, 0]))
        
    # Set the input state 
    input_state = pcvl.BasicState([1,0,1,0,1,0,1,0])
    proc.with_input(input_state)

    # Apply the minimum photon filter after setting up the circuit
    proc.min_detected_photons_filter(minPhoton)

    pcvl.pdisplay(proc)
    print('M: input_state=%s, min_photons=%d' % (str(input_state), minPhoton),proc.name)
        
    sampler = Sampler(proc, max_shots_per_call=shots)

    resD=sampler.sample_count(shots)  # blocking call
    print('raw resD:');    print(resD)
    print('shots=%2g' % shots,proc.name)
    
    # Handle different performance metrics based on simulator type
    if isIdeal:
        # Local/ideal simulator
        print('perf: global=%.6f' % resD['global_perf'])
    else:
        # Remote/noisy simulator
        print('perf: physical=%.6e, logical=%.6f' % (resD['physical_perf'], resD['logical_perf']))

    print('counts:', resD['results'])
    

