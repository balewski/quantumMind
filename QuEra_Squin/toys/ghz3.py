#!/usr/bin/env python3
"""Bell-state sampler using Bloqade's Gemini Cirq noise pipeline.

Converts a SQUIN Bell circuit to Cirq, applies the selected Gemini one- or
two-zone noise model, loads the noisy circuit back into SQUIN, and samples it.
Because Gemini Cirq models do not currently inject atom loss, --atom_loss is
implemented as one explicit SQUIN qubit_loss checkpoint before measurement.
"""

import argparse
from typing import Any

from bloqade import squin
from bloqade.cirq_utils import emit_circuit, noise, load_circuit
from bloqade.types import MeasurementResult
from dump_noiseModel import (
    dump_noise_parameters,
    noise_model_name,
    select_noise_model,
    set_loss_parameters,
)
from kirin.dialects import ilist
from Util_Squin import (
    get_lost_measurement_result,
    print_counts,
    print_kernel_circuit,
    print_loss_summary,
    run_kernel_shots,
    split_erasure_counts,
)


def generate_bell_program():
    @squin.kernel
    def bell_prog() -> ilist.IList[MeasurementResult, Any]:
        q = squin.qalloc(2)
        squin.h(q[0])
        squin.cx(q[0], q[1])
        return squin.broadcast.measure(q)
    return bell_prog

def apply_gemini_noise(kernel_prog: Any, zone_type: int, atom_loss: float, verb: int):
    """Transpilation pipeline as shown in your heuristic noisy simulation slide."""
    # 1. Setup the hardware-specific noise model.
    # Bloqade's Gemini Cirq models currently do not inject atom loss, so
    # atom_loss is also applied explicitly as a SQUIN qubit_loss channel below.
    noise_model = select_noise_model(zone_type)
    if atom_loss != 0.0:
        noise_model = set_loss_parameters(noise_model, atom_loss)
    if verb > 1:
        print(f"\n{noise_model_name(zone_type)} parameters:")
        dump_noise_parameters(noise_model)
    
    # 2. Export Squin kernel to Cirq
    cirq_ideal = emit_circuit(kernel_prog, ignore_returns=True) #
    if verb > 0:
        print("\nIdeal Cirq circuit before noise:")
        print(cirq_ideal)
    
    # 3. Add noise using QuEra's Gemini model
    cirq_noisy = noise.transform_circuit(cirq_ideal, model=noise_model) #
    
    # 4. Cirq back to a Squin kernel. With return_register=True this returns
    # qubits, so wrap it in a measured SQUIN program before sampling shots.
    noisy_prep = load_circuit(cirq_noisy, return_register=True, kernel_name="bell_gemini_prep") #

    @squin.kernel
    def noisy_bell_prog(atom_loss: float) -> ilist.IList[MeasurementResult, Any]:
        q = noisy_prep()
        squin.broadcast.qubit_loss(atom_loss, q)
        return squin.broadcast.measure(q)

    return noisy_bell_prog, (atom_loss,)

def main() -> None:
    parser = argparse.ArgumentParser()
    prs = parser.add_argument
    prs("--shots", type=int, default=1000)
    prs("-v", "--verb", type=int, default=1, help="verbosity")
    prs("--zone_type", type=int, choices=(1, 2), default=1, help="Gemini zone architecture: 1 or 2")
    prs("--atom_loss", type=float, default=0.05, help="if nonzero, set all loss parameters to this value")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

 
    # Define and transpile the circuit with noise
    ideal_prog = generate_bell_program()
    noisy_prog, noisy_args = apply_gemini_noise(
        ideal_prog, args.zone_type, args.atom_loss, args.verb
    )
    print_kernel_circuit(noisy_prog, noisy_args, args.verb)

    # Run the noisy simulation
    # Thrust 1 & 2: Sampling provides the realistic datasets needed for ML training [cite: 75, 81]
    lost_result = get_lost_measurement_result()
    if args.atom_loss > 0:
        if lost_result is None:
            print("erasure readout: this PyQrack install does not expose a Lost result")
        else:
            print("erasure readout: lost atoms are reported as e")
    counts = run_kernel_shots(
        noisy_prog, noisy_args, args.shots, loss_m_result=lost_result
    )
    
    print(f"Noisy Bell-state results ({noise_model_name(args.zone_type)}, {args.shots} shots)")
    counts_no_erasure, counts_with_erasure = split_erasure_counts(counts)
    shots_no_erasure = sum(counts_no_erasure.values())
    shots_with_erasure = sum(counts_with_erasure.values())
    print_loss_summary(counts)
    print(f"shots requested: {args.shots}")
    print(f"shots without erasure: {shots_no_erasure}")
    print(f"shots with erasure: {shots_with_erasure}")
    print(f"\nCounts without erasure {shots_no_erasure}:")
    print_counts(counts_no_erasure)
    print(f"\nCounts with erasure {shots_with_erasure}:")
    print_counts(counts_with_erasure)

if __name__ == "__main__":
    main()
