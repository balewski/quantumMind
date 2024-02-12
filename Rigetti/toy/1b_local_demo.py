#!/usr/bin/env python3
# less trivial demo, shows synchronous
from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, MEASURE
from functools import reduce
from pyquil.api import QPUCompiler

#...!...!....................
def shotM_2_counts(shot_meas):
    results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
    counts = dict(zip(results,[results.count(i) for i in results]))
    return counts


# Build a program
p = Program()
p += H(0)
p += CNOT(0, 10)
ro = p.declare("ro", "BIT", 2)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])
p.wrap_in_numshots_loop(10) # to perform multi-shot execution
qprg=p
print('M:p =\n',qprg)

# fire once the servers: /src/pyquil/entrypoint.sh 
# Get a quantum virtual machine (simulator)
device_name ="10q-qvm"
qback = get_qc(device_name)

# Execute the program synchronously
#  run this program on the Quantum Virtual Machine (QVM)

# Execute the program synchronously
shot_meas = qback.run(qprg).readout_data.get("ro")
print('M:shot_meas1',shot_meas)
counts= shotM_2_counts(shot_meas)
print('M:counts1',device_name,counts)


# repeat the same execution on simulated HW
device_name = 'Aspen-11'
qback2 = get_qc(device_name, as_qvm=True)
qprgT2 = qback2.compile(qprg)
print('M:qprgT2, device=%s\n'%device_name,qprgT2)
shot_meas = qback2.run(qprgT2).readout_data.get("ro")
print('M:shot_meas2',shot_meas)
counts= shotM_2_counts(shot_meas)
print('M:counts2',device_name,counts)

if isinstance(qback2.compiler, QPUCompiler):
    # Working with a QPU - refresh calibrations
    qback2_calib_prog=qback2.compiler.get_calibration_program(force_refresh=True)
    print('got calib', type(qback2_calib_prog))
else:
    print('M:no calib found for:',device_name)


pT3 = qback2.compiler.quil_to_native_quil(qprg, protoquil=True)
print('\nM:to_native',pT3)
#?print(pT3.metadata)



print('M:done')


