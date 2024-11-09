
import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator
import mapomatic as mm
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

nq=8
qc = QuantumCircuit(nq)
qc.h(0)
for i in range(1,nq):  qc.cx(0,i)
qc.measure_all()

print(qc)
service = QiskitRuntimeService()
noisy_backend = service.backend('ibm_torino')
backend = AerSimulator.from_backend(noisy_backend)
print('\n 1st pass, to convert circuit to naitive gates for ',backend.name)

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

qcT1 = pm.run(qc)
print(qcT1.draw('text', idle_wires=False))
layout0 = qcT1._layout.final_index_layout(filter_ancillas=True)
print(' initial phys layout:%s'%(layout0))

# remove idle qubits
qcT2 = mm.deflate_circuit(qcT1)
score0 = mm.evaluate_layouts(qc, [layout0], backend)
print('score0:',score0)
