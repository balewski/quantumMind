#!/usr/bin/env python3

from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager

#  ibmq_data/inc_advection_solver_aug21/inc_advection_solver_set300.qpy 

#coreN = 'advection_solver_set1'
#inpF = '../ibmq_data/inc_advection_solver_aug16/ibmq_qc_inc_%s.qpy' % coreN
inpF='inc_advection_solver_set300.qpy'
with open(inpF, 'rb') as fd:
    qcL = qpy.load(fd)


# assemble circuit
nq = qcL[0].num_qubits
print('nq=', nq)
qc = QuantumCircuit(nq)
for i in range(2):
    qc.append(qcL[i], range(nq))
qc.measure_all()
print(qc)
print(qc.decompose())

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()

backN = 'ibm_kingston'
backend = service.backend(backN, use_fractional_gates=True)  # enable rzz gates
print('use true HW backend =', backN)

# backend and optimization level.
print("Generating preset pass manager...")
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)

# Now, we run the circuit through the pass manager to get the transpiled circuit.
print("Running circuit through the pass manager...")
qcT = pass_manager.run(qc)
# --- END MODIFIED SECTION ---

print('Transpiled, ops:', qcT.count_ops())

#print(qcT)

sampler = Sampler(mode=backend)
# SamplerV2 expects an iterable of circuits. A list is a common way to provide it.
job = sampler.run([qcT], shots=100)

print(f"Job submitted with ID: {job.job_id()}")
