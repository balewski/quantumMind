#!/usr/bin/env python3
# exercise GHZ circuit with varied number of qubits
from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, MEASURE
from functools import reduce
from pyquil.api import QPUCompiler
from pprint import pprint


#...!...!....................
def shotM_2_counts(shot_meas):
    results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
    counts = dict(zip(results,[results.count(i) for i in results]))
    counts_ord = sorted(counts.items(),  key=lambda item: item[1], reverse=True)
    return counts_ord

#...!...!....................
def ghz_circ(nq=3):    # Build a program
    p = Program()
    ro = p.declare("ro", "BIT", nq)
    
    p += H(0)
    for j in range(1,nq):
        p += CNOT(0, j)
        
    for j in range(0,nq):
        p += MEASURE(j, ro[j])

    return p

#=================================
#=================================
#  M A I N
#=================================
#=================================
nqub=3; shots=200
print('\n\n ******  NEW circuit : GHZ  ****** ',nqub,shots)
qprg=ghz_circ(nqub)
qprg.wrap_in_numshots_loop(shots) # to perform multi-shot execution

print('M:p =\n',qprg)

# fire once the servers: /src/pyquil/entrypoint.sh 
# Get a quantum virtual machine (simulator)
device_name ="%dq-qvm"%nqub
qback = get_qc(device_name)

# Execute the program synchronously
#  run this program on the Quantum Virtual Machine (QVM)

# Execute the program synchronously
shot_meas = qback.run(qprg).readout_data.get("ro")
#print('M:shot_meas1',shot_meas)
counts= shotM_2_counts(shot_meas)
print('M:counts1',device_name,counts)


# repeat the same execution on simulated HW
device_name = 'Aspen-11'
qback2 = get_qc(device_name, as_qvm=True)
qprgT2 = qback2.compile(qprg)
print('\nM:qprgT2, device=%s\n'%device_name,qprgT2)
shot_meas = qback2.run(qprgT2).readout_data.get("ro")
#print('M:shot_meas2',shot_meas)
counts= shotM_2_counts(shot_meas)
print('M:counts2',device_name,counts)

print('\nM: switch to noisy device')
device_name ="%dq-noisy-qvm"%nqub
qback3 = get_qc(device_name)
shot_meas = qback3.run(qprg).readout_data.get("ro")
counts= shotM_2_counts(shot_meas)
print('M:counts3',device_name, 'nsol=',len(counts)); pprint(counts[:10])




print('M:done')


