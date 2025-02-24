#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
import perceval as pcvl
import re

#...!...!.................... 
def dualrail_to_bitstring(key):
    """
    Convert a dual-rail encoded key to a qubit bitstring.
    Returns 'bad' if the input is invalid.
    """
    # Extract numbers from the BasicState key, e.g., "|1,0,1,0>" -> [1, 0, 1, 0]
    key=str(key) # in case input is exqalibur.FockState
    modes = list(map(int, re.findall(r'\d+', key)))
    
    # Check for even number of modes (required for dual-rail)
    if len(modes) % 2 != 0:
        return 'bad'
    
    # Convert modes to bitstring
    bitstring = ''
    for i in range(0, len(modes), 2):
        # Each qubit is represented by a pair of modes
        if modes[i] == 1 and modes[i+1] == 0:
            bitstring += '0'  # Photon in first mode → |0>
        elif modes[i] == 0 and modes[i+1] == 1:
            bitstring += '1'  # Photon in second mode → |1>
        else:
            return 'bad'  # Invalid dual-rail pattern
    
    return bitstring

#...!...!.................... 
def bitstring_to_dualrail(bitstring):
    """
    Convert a qubit bitstring to a dual-rail encoded BasicState.
    Returns 'bad' if the input is invalid.
    """
    # Check for valid input (only '0' and '1' allowed)
    if not all(b in '01' for b in bitstring):
        return 'bad'
    
    # Convert each bit to dual-rail pair
    dualrail_pairs = []
    for b in bitstring:
        if b == '0':
            dualrail_pairs.append('1,0')  # |0> → |1,0>
        elif b == '1':
            dualrail_pairs.append('0,1')  # |1> → |0,1>
    
    # Join pairs and format as BasicState
    return '|' + ','.join(dualrail_pairs) + '>'



#...!...!.................... 
def dualRailState_to_bitstring(basic_state):
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

#...!...!.................... 
def bitstring_to_dualRailState(bits):
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




#...!...!.................... 

#...!...!.................... 

#...!...!.................... 

#...!...!.................... 


#=================================
#   U N I T   T E S T
#=================================

if __name__=="__main__":
    print('testing Perceval utility functions')
    print('ver:',pcvl.__version__)
    
    # Example usage
    example_state = pcvl.BasicState([1, 0, 0, 1])  # Should represent qubit state |0⟩ |1⟩
    qubit_state = dualRailState_to_bitstring(example_state)

    print("\nINPUT BasicState:", example_state)
    print("Qubit State as Bitstring:", qubit_state)

    # Example usage
    bitstring = '0101'  # Should represent BasicState |1,0,0,1,1,0,0,1>
    basic_state = bitstring_to_dualRailState(bitstring)

    print("\nINPUT Bitstring:", bitstring)
    print("BasicState:", basic_state)

    
    print('\nbitstring conversion')
    print(dualrail_to_bitstring('|1,0,1,0>'))  # Output: '01'
    print(dualrail_to_bitstring('|0,1,1,0>'))  # Output: '11'
    print(dualrail_to_bitstring('|1,1,0,0>'))  # Output: 'bad' (Invalid dual-rail state)
    print(dualrail_to_bitstring('|1,0,1>'))    # Output: 'bad' (Odd number of modes)

    print('\nbitstring conversion in reverse')
    print(bitstring_to_dualrail('01'))  # Output: '|1,0,0,1>'
    print(bitstring_to_dualrail('11'))  # Output: '|0,1,0,1>'
    print(bitstring_to_dualrail('10'))  # Output: '|0,1,1,0>'
    print(bitstring_to_dualrail('2'))   # Output: 'bad' (Invalid character)
    print(bitstring_to_dualrail('0a1')) # Output: 'bad' (Invalid character)
