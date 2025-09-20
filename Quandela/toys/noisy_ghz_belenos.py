#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

'''

import perceval as pcvl
from perceval.algorithm import Sampler
cnotH = pcvl.catalog["heralded cnot"].build_processor()

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    num_qubit=3
    nMode=2*num_qubit
    minPhoton=num_qubit 
    
    #proc = pcvl.Processor("SLOS",nMode) ; shots=10_000
    proc = pcvl.RemoteProcessor("sim:belenos",m=nMode) ; shots=10_000_000_000_000
    proc.min_detected_photons_filter(minPhoton)
    proc.add(0, pcvl.BS.H())
    proc.add(0, cnotH)
    proc.add(2, cnotH)
    proc.with_input(pcvl.BasicState([1,0,1,0,1,0]))
    pcvl.pdisplay(proc)
   
    print('M: using  processor',proc.name)
    
    sampler = Sampler(proc, max_shots_per_call=shots)
  
    resD=sampler.sample_count(shots)
       
    print('counts:', resD['results'])
    print('shots=%2g' % shots,proc.name)


