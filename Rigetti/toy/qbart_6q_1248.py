#!/usr/bin/env python3

from pyquil import Program, get_qc
from pyquil.gates import RY, CNOT, MEASURE, H
from functools import reduce
import numpy as np

shots = 1024

p = Program()

ro = p.declare('ro', memory_type='BIT', memory_size=6)

p.inst(RY(np.pi / 4, 0))
p.inst(RY(np.pi / 4, 1))
p.inst(RY(np.pi / 4, 2))
p.inst(RY(np.pi / 4, 3))
p.inst(H(4))
p.inst(H(5))
p.inst(CNOT(4, 3))
p.inst(CNOT(5, 2))
p.inst(CNOT(4, 1))
p.inst(CNOT(5, 0))
p.inst(RY(np.pi / 4, 0))
p.inst(RY(-np.pi / 4, 1))
p.inst(RY(-np.pi / 4, 2))
p.inst(RY(-np.pi / 4, 3))
p.inst(CNOT(5, 3))
p.inst(CNOT(4, 2))
p.inst(CNOT(5, 1))
p.inst(CNOT(4, 0))
p.inst(RY(np.pi / 4, 0))
p.inst(RY(-np.pi / 4, 1))
p.inst(RY(-np.pi / 4, 2))
p.inst(RY(np.pi / 4, 3))
p.inst(CNOT(4, 3))
p.inst(CNOT(5, 2))
p.inst(CNOT(4, 1))
p.inst(CNOT(5, 0))
p.inst(RY(np.pi / 4, 0))
p.inst(RY(np.pi / 4, 1))
p.inst(RY(np.pi / 4, 2))
p.inst(RY(-np.pi / 4, 3))
p.inst(CNOT(5, 3))
p.inst(CNOT(4, 2))
p.inst(CNOT(5, 1))
p.inst(CNOT(4, 0))
p.inst(MEASURE(0, ro[0]))
p.inst(MEASURE(1, ro[1]))
p.inst(MEASURE(2, ro[2]))
p.inst(MEASURE(3, ro[3]))
p.inst(MEASURE(4, ro[4]))
p.inst(MEASURE(5, ro[5]))

p.wrap_in_numshots_loop(shots)
device_name = '6q-qvm'
qc = get_qc(device_name, as_qvm=True)
results_list = qc.run(p).readout_data.get("ro")
results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), results_list))
counts = dict(zip(results,[results.count(i) for i in results]))
print(counts)

#...........  emulate ASPEN-11
device_name = 'Aspen-11'
qc = get_qc(device_name, as_qvm=True)
pT = qc.compile(p)
print('M:pT, device=%s\n'%device_name,pT,type(pT))

# Execute the program synchronously
result = qc.run(pT)
shot_meas = result.readout_data.get('ro')
print('M:shot_meas',shot_meas)
results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
counts = dict(zip(results,[results.count(i) for i in results]))
print('M:counts',counts)


print('M:done')
