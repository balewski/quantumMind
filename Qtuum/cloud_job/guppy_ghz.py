#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
from time import time
from collections import Counter

# Explicit Guppy imports
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import qubit, h, cx, measure_array

def generate_ghz_program(num_qubits: int):
    """
    Factory function to build a Guppy program that prepares an N-qubit GHZ state.
    Uses 'comptime' to evaluate the Python integer into a static Guppy array size.
    """
    
    @guppy
    def main_ghz() -> None:
        # 1. Allocate an array of N qubits. comptime() injects the Python variable.
        qs = array(qubit() for _ in range(comptime(num_qubits)))
        
        # 2. Apply Hadamard to the first qubit
        h(qs[0])
        
        # 3. Entangle the rest in a chain. 
        # The loop range is also evaluated at compile-time to unroll the gates.
        for i in range(comptime(num_qubits - 1)):
            cx(qs[i], qs[i + 1])
            
        # 4. Measure the entire array and store it in result register 'c'
        result("c", measure_array(qs))

    return main_ghz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guppylang N-Qubit GHZ Example")
    parser.add_argument("-q", "--numQubits", type=int, default=5, help="Number of qubits for the GHZ state")
    parser.add_argument("-n","--numShots", type=int, default=1000, help="Number of shots to execute")
    parser.add_argument("-b", "--backendType", type=int, default=1, help="0: Ideal Statevector, 1: Fast Stabilizer (Stim)")
    args = parser.parse_args()
    
    print(f'--- 1. Compiling {args.numQubits}-qubit Guppy function to HUGR ...')
    t0 = time()
    
    # Generate the dynamically sized Guppy function
    ghz_prog = generate_ghz_program(args.numQubits)
    
    # Static type check and compile
    ghz_prog.check()
    hugrPackage = ghz_prog.compile()
    
    t1 = time()
    print('success')
    print(f'elaT={t1-t0:.1f} sec...')
    
    print('--- 2. Setting up Selene Emulator ...')
    t0 = time()
    
    # We must inform the emulator of the total qubit allocation needed
    emuBase = ghz_prog.emulator(n_qubits=args.numQubits)
    
    if args.backendType == 0:
        print("Selected: Ideal Statevector Simulator")
        runner = emuBase.statevector_sim()
    else:
        print("Selected: Fast Stabilizer Simulator (Stim)")
        runner = emuBase.stabilizer_sim()
        
    runner = runner.with_shots(args.numShots)
    
    t1 = time()
    print(f'elaT={t1-t0:.1f} sec...')
    
    print('--- 3. Executing Circuit ...')
    t0 = time()
    
    simResults = runner.run()
    
    t1 = time()
    print('success')
    print(f'elaT={t1-t0:.1f} sec...')

    print('--- 4. Processing Output ...')
    t0 = time()
    
    shots_list = simResults.collated_shots()
    print(f"\nMeasurement results over {args.numShots} shots:")
    
    if len(shots_list) > 0:
        print(f"Debug - Raw first shot looks like: {shots_list[0]}")
    
    freqD = Counter()
    for shot in shots_list:
        # Get 'c' register - typically nested as [[b0, b1, ...]]
        bits = shot.get("c", [[]])[0]
        bitstring = "".join(map(str, map(int, bits)))
        freqD[bitstring] += 1
        
    print(f"\nOutcome frequencies (expecting mostly all 0s and all 1s):")
    for k, v in freqD.most_common():
        print(f"  {k} : {v} shots")
    
    t1 = time()
    print(f'elaT={t1-t0:.1f} sec...')
