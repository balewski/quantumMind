"""Shared SQUIN toy helpers.

Contains generic bitstring conversion, circuit printing, PyQrack one-shot
sampling, erasure-count splitting, probability validation, and count formatting.
"""

import inspect
import time
from collections import Counter
from typing import Any

from bloqade.cirq_utils import emit_circuit
from bloqade.pyqrack import DynamicMemorySimulator
from bloqade.types import MeasurementResult
from kirin.dialects import ilist


def outcome_to_bitstring(outcome: ilist.IList[MeasurementResult, Any]) -> str:
    bits = []
    for measurement in outcome:
        is_lost = getattr(measurement, "is_lost", False)
        if callable(is_lost):
            is_lost = is_lost()
        if is_lost:
            bits.append("e")
            continue

        value = getattr(measurement, "value", measurement)
        value_name = getattr(value, "name", "")
        name = getattr(measurement, "name", value_name).lower()
        if "lost" in name or "loss" in name or "erasure" in name:
            bits.append("e")
            continue

        if value == 0:
            bits.append("0")
        elif value == 1:
            bits.append("1")
        elif name.endswith("zero"):
            bits.append("0")
        elif name.endswith("one"):
            bits.append("1")
        else:
            bits.append(str(measurement))
    return "".join(bits)


def print_kernel_circuit(kernel: Any, kernel_args: tuple[Any, ...], verb: int) -> None:
    if verb > 1:
        kernel.print()
    elif verb == 1:
        print(emit_circuit(kernel, args=kernel_args, ignore_returns=True))


def run_kernel_shots(
    kernel: Any,
    kernel_args: tuple[Any, ...],
    shots: int,
    *,
    loss_m_result: Any | None = None,
) -> Counter[str]:
    if loss_m_result is None:
        emulator = DynamicMemorySimulator()
    else:
        emulator = DynamicMemorySimulator(loss_m_result=loss_m_result)
    task = emulator.task(kernel, args=kernel_args)
    start = time.perf_counter()
    counts = Counter(outcome_to_bitstring(task.run()) for _ in range(shots))
    elapsed = time.perf_counter() - start
    print(f"sampling of {shots} shots took {elapsed:.1f} sec")
    return counts


def get_lost_measurement_result() -> Any | None:
    param = inspect.signature(DynamicMemorySimulator).parameters.get("loss_m_result")
    if param is None or param.default is inspect.Parameter.empty:
        return None

    measurement_type = type(param.default)
    return getattr(measurement_type, "Lost", None)


def validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def print_loss_summary(counts: dict[str, int]) -> None:
    loss_shots = sum(count for bitstring, count in counts.items() if "e" in bitstring)
    lost_qubits = sum(bitstring.count("e") * count for bitstring, count in counts.items())
    print(f"loss shots: {loss_shots}")
    print(f"lost qubit measurements: {lost_qubits}")


def split_erasure_counts(counts: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    no_erasure = {}
    with_erasure = {}
    for bitstring, count in counts.items():
        if "e" in bitstring:
            with_erasure[bitstring] = count
        else:
            no_erasure[bitstring] = count
    return no_erasure, with_erasure


def print_counts(counts: dict[str, int]) -> None:
    for bitstring, count in sorted(counts.items()):
        print(f"{bitstring}: {count:5d} shots")
