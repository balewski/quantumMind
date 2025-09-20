#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Density matrix simulation with noise models extracted from real IBM quantum devices

Example of density matrix computation using Qiskit 1.2

    https://docs.quantum.ibm.com/guides/build-noise-models
    https://docs.quantum.ibm.com/guides/build-noise-models#initializing-a-noise-model-from-a-backend
"""
from qiskit import  QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from bigEndianUnitary import print_complex_nice_matrix

from pprint import pprint

# iSWAP matrix operator
iswap_op = Operator([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]])
# Using operators in circuits
nq=2;nb=2
qc = QuantumCircuit(2)

# Add gates
qc.sdg(1)
qc.h(1)
qc.sdg(0)
qc.unitary(iswap_op, [0, 1], label='iswap')
qc.sdg(0)
qc.save_density_matrix()
qc.sdg(0)
qc.unitary(iswap_op, [0, 1], label='iswap')
qc.s(1)

print(qc)

backend1 = AerSimulator(method="density_matrix")
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend1.name))

job = backend1.run(qc)
result = job.result() 
#!pprint(result)
#1print('ideal out density matrix:',result.data())
rho1=result.data()['density_matrix']
print_complex_nice_matrix(rho1,'ideal out density matrix')


print('\n repeat on  fake backend ...')
service = QiskitRuntimeService(channel="ibm_quantum")

backName='ibm_torino'  # EPLG=0.7%
backName='ibm_kyiv'    # EPLG=1.4%
backName='ibm_nazca'   # EPLG=3.2%

noisy_backend = service.backend(backName)
print('use noisy_backend =', noisy_backend.name )
# Create noisy simulator backend
noise_model = NoiseModel.from_backend(noisy_backend)
backend2 = AerSimulator(method="density_matrix", noise_model=noise_model)


pm = generate_preset_pass_manager(optimization_level=3, backend=backend2)
qcT = pm.run(qc)

#qcT= transpile(qc, backend2)
print('transpiled for',backend2)
print(qcT.draw('text', idle_wires=False))


job2 = backend2.run(qcT)
result2 = job2.result()
rho3=result2.data()['density_matrix']
print_complex_nice_matrix(rho3,'noisy out density matrix')



