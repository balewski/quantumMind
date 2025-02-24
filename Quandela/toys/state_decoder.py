#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
import perceval as pcvl

def basicstate_to_qubit(basic_state):
    """
    Convert a Perceval BasicState to a qubit state as a bitstring,
    assuming dual-rail encoding.
    """
    # Ensure even number of modes (dual-rail pairs)
    if len(basic_state) % 2 != 0:
        raise ValueError("BasicState length must be even for dual-rail encoding.")

    # Convert to qubit state as a bitstring
    qubit_state = ''
    for i in range(0, len(basic_state), 2):
        # Each qubit is represented by a pair of modes
        if basic_state[i] == 1 and basic_state[i+1] == 0:
            qubit_state += '0'  # Photon in first mode → |0>
        elif basic_state[i] == 0 and basic_state[i+1] == 1:
            qubit_state += '1'  # Photon in second mode → |1>
        else:
            raise ValueError("Invalid dual-rail state: each qubit must have exactly one photon in its pair.")
    
    return qubit_state


def qubit_to_basicstate(bits):
    """
    Convert a qubit state bitstring to a Perceval BasicState,
    assuming dual-rail encoding.
    """
    # Check input validity
    if not all(b in '01' for b in bits):
        raise ValueError("Input must be a bitstring containing only '0' and '1'.")

    # Convert bitstring to dual-rail BasicState
    state = []
    for b in bits:
        if b == '0':
            state.extend([1, 0])  # |0> → |1, 0>
        elif b == '1':
            state.extend([0, 1])  # |1> → |0, 1>
    
    return pcvl.BasicState(state)

# Example usage
example_state = pcvl.BasicState([1, 0, 0, 1])  # Should represent qubit state |0⟩ |1⟩
qubit_state = basicstate_to_qubit(example_state)

print("\nINPUT BasicState:", example_state)
print("Qubit State as Bitstring:", qubit_state)

# Example usage
bitstring = '0101'  # Should represent BasicState |1,0,0,1,1,0,0,1>
basic_state = qubit_to_basicstate(bitstring)

print("\nINPUT Bitstring:", bitstring)
print("BasicState:", basic_state)
