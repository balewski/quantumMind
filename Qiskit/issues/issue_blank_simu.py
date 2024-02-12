#!/usr/bin/env python3
''' problem: ??

'''
import qiskit as qk

backend = qk.Aer.get_backend('aer_simulator')

# -------Create a Quantum Circuit 
circ = qk.QuantumCircuit(3,3)
circ.h(0)
circ.cx(0, 1) 
circ.delay(400)
circ.cx(0, 2)
circ.barrier()
circ.measure(0,0)
print(circ)
print('\n transpile\n')
circT = qk.transpile(circ, backend=backend, optimization_level=1, seed_transpiler=111, basis_gates=['p','sx','cx'])
print(circT)

job =  backend.run(circT,shots=1000)
jid=job.job_id()

counts = job.result().get_counts(0)
print(counts)
print('M:ok')

