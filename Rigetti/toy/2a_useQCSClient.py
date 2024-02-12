#!/usr/bin/env python3
# less trivial demo, shows synchronous and asynchronous execution
from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, MEASURE
from pyquil.api import QPUCompiler
from functools import reduce
# access credentials
from pyquil.api import QCSClientConfiguration
qcsCli = QCSClientConfiguration.load()
print(qcsCli)


# Build a program
p = Program()
p += H(0)
p += CNOT(0, 10)
ro = p.declare("ro", "BIT", 2)
p += MEASURE(0, ro[0])
p += MEASURE(1, ro[1])
p.wrap_in_numshots_loop(10) # to perform multi-shot execution
print('M:p =\n',p)

device_name = 'Aspen-11'
device_name ="10q-qvm"
backend = get_qc(device_name, client_configuration=qcsCli)  
if isinstance(backend.compiler, QPUCompiler) and 0:
    # Working with a QPU - refresh calibrations
    qvmA11.compiler.get_calibration_program(force_refresh=True)

result = backend.run(p)

# Execute the program synchronously
#  run this program on the Quantum Virtual Machine (QVM)
pT = backend.compile(p)
print('M:pT, device=%s\n'%device_name,pT)

# Execute the program synchronously
result = backend.run(pT)
shot_meas = result.readout_data.get('ro')
print('M:shot_meas',shot_meas)
results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
counts = dict(zip(results,[results.count(i) for i in results]))
print('M:counts',counts)


print('M:done')
