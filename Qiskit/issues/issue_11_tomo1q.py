#!/usr/bin/env python3
''' problem: ??

'''
import qiskit as qk
from qiskit_experiments.library import StateTomography

backend = qk.Aer.get_backend('aer_simulator')

# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(1)
circ.rx(1.3,0)
print(circ)
qstexp1 = StateTomography(circ)
qcL=qstexp1.circuits()

print('nqc',len(qcL))
for qc in qcL:   print('ideal',qc.name); print(qc)


qcLT = qk.transpile(qcL, backend, basis_gates=['p','sx','cx'] , seed_transpiler=111,optimization_level=1)

job = backend.run(qcLT, shots=1000)
print("M: sim circ on  backend=%s"%(backend))
result = job.result()

for qc in qcLT:
    counts = result.get_counts(qc)
    print('simu circ:%s, counts:'%qc.name,end=''); print(counts)
    print(qc.draw(output="text", idle_wires=False))
