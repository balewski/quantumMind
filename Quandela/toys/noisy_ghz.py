#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Hello,

sim:ascella is meant to reproduce the behaviour of qpu:ascella using simulation. As such, you cannot give more photons as inputs than the actual number of input ports in the hardware. For Ascella, the cloud indicates that the maximum number of photons is 6.

In your circuit, you use 3 photons for your qubits and an additionnal 4 photons for the heralds of the heralded cnots, as stated in the error message. If you want to avoid this problem, you can use postselected cnots that don’t add more photons (beware you will need to change your minPhoton to 3). Your previous jobs on qpu:ascella worked because you only had two qubits and one cnot, so 4 photons.

Due to the existence of postselections in the postselected cnots, you can’t add several of them manually on overlapping modes. To turn around this, you can use one heralded cnot then one postprocessed cnot, or remove the postselections and put it back at the end, or use a converter that does that for you (as well as not using postselected cnot if this would lead to wrong results, but this is not the case for a simple GHZ state generator).

I hope this will help you,

Regards,
	Aubaert

'''
import perceval as pcvl
from perceval.algorithm import Sampler
cnotH = pcvl.catalog["heralded cnot"].build_processor()
cnotP = pcvl.catalog['postprocessed cnot'].build_processor()
num_qubit=3
nMode=2*num_qubit
minPhoton=num_qubit + 2  # because only 1  herald-CNOT is used 
isIdeal=True
if isIdeal:
    proc = pcvl.Processor("SLOS",nMode)
else:
    proc = pcvl.RemoteProcessor("sim:ascella",m=nMode)
proc.min_detected_photons_filter(minPhoton)
proc.add(0, pcvl.BS.H())
proc.add(0, cnotH)
proc.add(2, cnotP)
proc.with_input(pcvl.BasicState([1,0,1,0,1,0]))
pcvl.pdisplay(proc)
#1pcvl.pdisplay(proc, output_format=pcvl.Format.LATEX)
shots=100_000_000
sampler = Sampler(proc, max_shots_per_call=shots)
if isIdeal:
    resD=sampler.sample_count(shots)
else:
    job = sampler.sample_count.execute_async(shots)
    resD=job.get_results()
print('perf:',resD['physical_perf'],resD['logical_perf'])
print('counts:',resD['results'])
print(resD)
