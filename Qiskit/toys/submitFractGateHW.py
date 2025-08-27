#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler 

#................
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
def pm_transpile(backend,optimization_level=3,seed_transpiler=42):
    print("Generating preset pass manager...")
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, seed_transpiler=seed_transpiler)

    # FoldRzzAngle should be applied at post_optimisation to work w/ fractional gates, Daito
    pm.post_optimization = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=backend.target),
            RemoveIdentityEquivalent(target=backend.target),
        ]
    )
 
    qcT = pm.run(qc)
    return qcT
#................
    
inpF='inc_advection_solver_set400.qpy'

with open(inpF, 'rb') as fd:
    qcL = qpy.load(fd)

# assemble circuit
nq=qcL[0].num_qubits
print('nq=',nq)
qc=QuantumCircuit(nq)
for i in range(2):
    qc.append(qcL[i],range(nq))
qc.measure_all()
print(qc)

print('M: activate QiskitRuntimeService() ...')
service = QiskitRuntimeService()

backN='ibm_kingston'
backend = service.backend(backN , use_fractional_gates=True)  #  enable rzz gates
print('use true HW backend =', backN) 
tseed=42
qcT=pm_transpile(backend,optimization_level=3,seed_transpiler=tseed)
physQ = qcT._layout.final_index_layout(filter_ancillas=True)
print('qcT qubits:',physQ)
print('Transpiled, ops:',qcT.count_ops())
#1print(qcT.draw('text', idle_wires=False))

sampler = Sampler(mode=backend)
job = sampler.run(tuple([qcT]),shots=10)

print('circ submitted')
