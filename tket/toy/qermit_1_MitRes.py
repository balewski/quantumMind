#!/usr/bin/env python3
# https://cqcl.github.io/Qermit/manual/manual_mitres.html

print('M:start...')

# ---- ideal simulator Aer
from qermit import MitRes, CircuitShots
from pytket import Circuit
from pytket.extensions.qiskit import AerBackend

mitres = MitRes(backend = AerBackend())
c = Circuit(2,2).H(0).Rz(0.25,0).CX(1,0).measure_all()
results = mitres.run([CircuitShots(Circuit = c, Shots = 50)])
print('Aer:',results[0].get_counts())
mitres.get_task_graph()  # no graphisc shows up???

# ------ It is also possible to import and use the MitTask objects directly.
from qermit.taskgraph import backend_handle_task_gen, backend_res_task_gen

sim_backend = AerBackend()
circuits_to_handles_task = backend_handle_task_gen(sim_backend)
handles_to_results_task = backend_res_task_gen(sim_backend)

print(circuits_to_handles_task)
print(handles_to_results_task)


handles = circuits_to_handles_task(([CircuitShots(Circuit = c, Shots = 50)],))
results = handles_to_results_task(handles)
print('use MitTask + Aer:',results[0][0].get_counts())
