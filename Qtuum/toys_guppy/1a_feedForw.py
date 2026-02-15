#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pprint import pprint 
import os, secrets
import qnexus as qnx
from time import time, sleep
from pytket.backends.backendresult import BasisOrder


# To implement this circuit, we define a Python function with the @guppy decorator. Since our circuit takes no inputs, the function does not have to have any parameters. Similarly, as the circuit prepares a single-qubit state, we must annotate the function with the corresponding return type. We also record the outcome of the mid-circuit measurement for later evaluation using result, as this will make it available later once we run the simulation.

@guppy
def simple_circuit() -> qubit:
    q1, q2 = qubit(), qubit()

    h(q1)
    cx(q1, q2)

    outcome = measure(q1)
    result("q1", outcome)

    if outcome:
        x(q2)

    return q2
print('check:',simple_circuit.check())

# For execution, we can write a function that invokes the circuit and consumes the produced qubit via a measurement. The outcome is recorded for later evaluation as well.



@guppy
def evaluate() -> None:
    q = simple_circuit()
    result("q2", measure(q))


# Finally, we can emulate our circuit implementation using the stabilizer simulator. Our program is executed for a single shot using the run method.

shots=5
emulator = evaluate.emulator(n_qubits=2).stabilizer_sim().with_seed(3)
sim_result = emulator.with_shots(shots).run()
#print(list(sim_result.results))

pprint(sim_result.results)

# Compile to HUGR IR
hugr_pkg = evaluate.compile()  # compile the full program
print('hugr done')

# Print circuit
print('\n--- Circuit ---')


if 1:
    myTag = '_' + secrets.token_hex(3)
    shots = 10

    #devName = "H2-1E" # Use for error-modelled emulation of H2.
    devName="Helios-1E" #full error modelling  , not working w/ TKet

    #myAccount = 'CSC641'
    myAccount = 'CHM170'
    project = qnx.projects.get_or_create(name="feb-guppy")
    qnx.context.set_active_project(project)

    print('define some TKET circ')

    # 3) Upload HUGR to Nexus
   
    t0 = time()
    ref = qnx.hugr.upload(
        hugr_package=hugr_pkg,
        name=f"jan-demo-{myTag}",
        description="2‑qubit Bell state demo from Guppy (HUGR).",
        project=project,
    )  # returns a HUGR reference usable in execute jobs [web:59][web:62]
    t1 = time()
    print('elaT=%.1f sec, uploaded, compiling ...'%(t1-t0))
    print('ref:',ref)
    devConf = qnx.QuantinuumConfig(device_name=devName, user_group=myAccount,max_cost=500, compiler_options={"max-qubits": 10})
    print('use devConf:', devConf)

    #.... execution     
    t0 = time()
    ref_exec = qnx.start_execute_job(programs=[ref], n_shots=[shots],
                                     backend_config=devConf, name="exec"+myTag)
    t1 = time()
    print('job submit elaT=%.1f, waiting for results ...'%(t1-t0))

    qnx.jobs.wait_for(ref_exec)
    results = qnx.jobs.results(ref_exec)
    t2 = time()
    print('execution finished, total elaT=%.1f\n'%(t2-t1))

    result = results[0].download_result()

    # Print per-shot results
    print(f'\n--- {len(result.results)} shots ---')
    for i, shot in enumerate(result.results):
        entries = {name: val for name, val in shot.entries}
        print(f'  shot {i}: {entries}')

    # Aggregate counts
    from collections import Counter
    counts = Counter()
    for shot in result.results:
        key = tuple(f'{name}={val}' for name, val in shot.entries)
        counts[key] += 1
    print('\nCounts:')
    for key, cnt in counts.most_common():
        print(f'  {", ".join(key)}: {cnt}')

    print('\ndone devName=', devName)
    print('status:', qnx.jobs.status(ref_exec))

    
