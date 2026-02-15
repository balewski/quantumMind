#!/usr/bin/env python3
"""
Feed-forward demo: Bell state with mid-circuit measurement and conditional X gate.
Compiles a Guppy circuit, runs a local stabilizer emulation,
then (optionally) submits to Quantinuum via Nexus.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
import secrets
from collections import Counter
from pprint import pprint
from time import time, sleep

import qnexus as qnx
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, h, measure, qubit, x


# ---- Guppy circuit definitions ----

@guppy
def my_circ_fn() -> qubit:
    """Prepare Bell pair, measure q1, conditionally flip q2."""
    q1, q2 = qubit(), qubit()
    h(q1)
    cx(q1, q2)
    outcome = measure(q1)
    result("q1", outcome)
    if outcome:
        x(q2)
    return q2


# ---- helpers ----

def build_evaluator(circuit_fn):
    """Build a guppy evaluator that measures the output qubit of *circuit_fn*."""
    @guppy
    def _evaluate() -> None:
        q = circuit_fn()
        result("q2", measure(q))
    return _evaluate


def run_emulation(guppy_prog, n_qubits: int = 2,
                  shots: int = 5, seed: int = 3):
    """Run local stabilizer emulation of *guppy_prog* and print results."""
    emulator = guppy_prog.emulator(n_qubits=n_qubits).stabilizer_sim().with_seed(seed)
    sim_result = emulator.with_shots(shots).run()
    print("\n--- Emulation results ---")
    pprint(sim_result.results)
    return sim_result


def submit_job(hugr_pkg, shots: int = 10,
               dev_name: str = "Helios-1E",
               account: str = "CHM170",
               project_name: str = "feb-guppy",
               max_cost: int = 500):
    """Step 1: Upload HUGR to Nexus, compile, and start execution job.

    Returns (ref_exec, tag) – the job reference and the unique tag.
    """
    tag = "_" + secrets.token_hex(3)

    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)

    # Upload HUGR
    t0 = time()
    ref = qnx.hugr.upload(
        hugr_package=hugr_pkg,
        name=f"jan-demo-{tag}",
        description="2-qubit Bell state demo from Guppy (HUGR).",
        project=project,
    )
    print("upload elaT=%.1f sec" % (time() - t0))

    dev_conf = qnx.QuantinuumConfig(
        device_name=dev_name,
        user_group=account,
        max_cost=max_cost,
        compiler_options={"max-qubits": 10},
    )
    print("devConf:", dev_conf)

    # Submit execution job
    t0 = time()
    ref_exec = qnx.start_execute_job(
        programs=[ref], n_shots=[shots],
        backend_config=dev_conf, name="exec" + tag,
    )
    print("job submit elaT=%.1f sec" % (time() - t0))
    print("job ref:", ref_exec)
    return ref_exec, tag


def retrieve_job(ref_exec):
    """Step 2: Wait for the job to finish, download result, and print metadata.

    Returns the downloaded result object.
    """
    t0 = time()
    print("waiting for job to complete ...")
    qnx.jobs.wait_for(ref_exec)
    wait_time = time() - t0

    status = qnx.jobs.status(ref_exec)
    results = qnx.jobs.results(ref_exec)
    result_data = results[0].download_result()

    print("\n--- Job metadata ---")
    print("  status :", status)
    print("  wait   : %.1f sec" % wait_time)
    print("  n_shots:", len(result_data.results))
    return result_data


def postprocess_shots(result_data):
    """Step 3: Print per-shot entries and aggregated counts."""

    print(f"\n--- {len(result_data.results)} shots ---")
    for i, shot in enumerate(result_data.results):
        entries = {name: val for name, val in shot.entries}
        print(f"  shot {i}: {entries}")

    # Aggregate counts
    counts = Counter()
    for shot in result_data.results:
        key = tuple(f"{name}={val}" for name, val in shot.entries)
        counts[key] += 1
    print("\nCounts:")
    for key, cnt in counts.most_common():
        print(f'  {", ".join(key)}: {cnt}')


# ---- main ----

def main():
    print("check:", my_circ_fn.check())
    shots=10
    # Build evaluator for the chosen circuit
    prog1 = build_evaluator(my_circ_fn)   

    # Local emulation
    run_emulation(prog1, n_qubits=2, shots=shots)

    # Compile to HUGR IR
    hugr_pkg1 = prog1.compile()
    print("hugr compiled")

    # Step 1 – submit
    ref_exec, tag = submit_job(hugr_pkg1, shots=shots)

    # Step 2 – retrieve & print metadata
    result_data = retrieve_job(ref_exec)

    # Step 3 – post-process shots
    postprocess_shots(result_data)


if __name__ == "__main__":
    main()