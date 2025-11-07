#!/usr/bin/env python3
"""
Test for ARM M3 docker image sinter+stim+pymatching compatibility.

Expected behavior:
- OLD image (unpatched sinter): AssertionError due to numpy.int64 in self.errors.
- FIXED image: Runs successfully and prints collected stats.
"""

import stim
import sinter

def main():
    print("=== Starting Sinter ARM Test ===")

    # Create a small surface code circuit for testing
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=3,
        rounds=10,
        after_clifford_depolarization=0.1  # equivalent to "noise"
    )

    tasks = [
        sinter.Task(
            circuit=circuit,
            json_metadata={"test": "arm_m3_patch"}
        )
    ]

    try:
        stats = sinter.collect(
            num_workers=2,         # triggers multiprocessing path
            tasks=tasks,
            decoders=['pymatching'],
            max_shots=100,
            max_errors=10
        )
        print("=== Test PASSED: collected stats ===")
        for s in stats:
            print(s)
    except Exception as e:
        print("=== Test FAILED ===")
        raise

if __name__ == "__main__":
    main()
