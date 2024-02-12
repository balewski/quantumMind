#!/usr/bin/env python3
# https://cqcl.github.io/Qermit/manual/manual_mitres.html

print('M:start...')
from qermit import MitTask

from pytket import Circuit
from pytket.backends import Backend
from typing import List

#  function that compiles Circuits to a given Backend.
def compile_circuits(backend: Backend, circuits: List[Circuit]) -> List[Circuit]:
   for c in circuits:
      backend.compile_circuit(c)
   return circuits


from qermit import CircuitShots
from typing import Tuple

#  more advanced function also includes shots list
def compile_circuit_shots(backend: Backend, circuit_shots: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
  compiled_circuit_shots = []
  for cs in circuit_shots:
        compiled_circuit = backend.get_compiled_circuit(cs.Circuit)
        compiled_circuit_shots.append((compiled_circuit, cs.Shots))
  return (compiled_circuit_shots,)


def backend_compile_circuit_shots_task_gen(
   backend: Backend
   ) -> MitTask:
   def compile_circuit_shots(obj, circuit_shots: List[CircuitShots]) -> Tuple[List[CircuitShots]]:
      compiled_circuit_shots = []
      for cs in circuit_shots:
            compiled_circuit = backend.get_compiled_circuit(cs.Circuit)
            compiled_circuit_shots.append((compiled_circuit, cs.Shots))
      return (compiled_circuit_shots,)

   return MitTask(
      _label="CompileCircuitShots", _n_in_wires=1, _n_out_wires=1, _method=compile_circuit_shots
   )

print(' ========== MAIN ======== ')
from pytket.extensions.qiskit import AerBackend
from qermit import MitRes

sim_backend = AerBackend()
mit_task = backend_compile_circuit_shots_task_gen(sim_backend)
print('start:',mit_task)
test_circuit_shots = [CircuitShots(Circuit = Circuit(2).CZ(0,1).measure_all(), Shots = 100)]
test_results = mit_task((test_circuit_shots,))
print('MitTask:',test_results)


mit_res = MitRes(sim_backend)
mit_res.prepend(mit_task)
mit_res.get_task_graph()  # no graphics?
