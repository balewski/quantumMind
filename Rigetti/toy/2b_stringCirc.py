#!/usr/bin/env python3
'''
defines Quil circ as string
'''

from pprint import pprint
from pyquil import Program, get_qc
from pyquil.quilbase import Declare
from pyquil.gates import CNOT, Z, MEASURE
from functools import reduce


#=================================
#=================================
#  M A I N
#=================================
#=================================

# Raw Quil with newlines
qprg=Program("DECLARE ro BIT[2]\nH 0\nY 1\nXY(pi) 11 12\nCNOT 1 4\nRY(0.34) 3\nMEASURE 1 ro[1]")


# undefined:  \nU3 (0.1,0.2,0.3) 7
# \nFENCE 0

qprg2 = Program(
    Declare("ro", "BIT", 2),
    Z(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
)

qprg.wrap_in_numshots_loop(5)
print('\n Quil program:')
print('M:qprg\n',qprg)
device_name = '6q-qvm'
qcom = get_qc(device_name, as_qvm=True)  #backend

shot_meas = qcom.run(qprg).readout_data.get("ro")
print('M:shot_meas',shot_meas)
results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
counts = dict(zip(results,[results.count(i) for i in results]))
print(counts)



print('M:done')
