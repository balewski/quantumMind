#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
This code uses manual post selection filtering.
See more details in toys/noisy_bell.ghz for automatic post selection filtering with 
 remote_simulator.set_postselection(PostSelect("[0, 1] == 1 and [2, 3] == 1 and ..."))
'''
from noisy_bell import monitor_async_job

import perceval as pcvl
from perceval.algorithm import Sampler

def print_results(results):
    """Print quantum states with bit string conversion and return bit-indexed dictionary
    
    Args:
        results: Dictionary of quantum state strings to counts
        
    Returns:
        Dictionary with bit strings as keys and counts as values
    """
    bit_results = {}
    
    if results:
        print('\nAccepted states:')
        for state, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            # Convert dual-rail to bit string
            from perceval.utils import BasicState
            photonic_state = BasicState(state)
            bit_string = ''
            # For each qubit pair, check which mode has the photon
            for i in range(0, len(photonic_state), 2):
                if photonic_state[i] == 1:
                    bit_string += '0'
                elif photonic_state[i+1] == 1:
                    bit_string += '1'
                else:
                    bit_string += '?'  # Invalid state
            
            print('%s  %s  %d' % (bit_string, state, count))
            bit_results[bit_string] = count
    else:
        print('counts: (no results)')
    
    return bit_results

def apply_postselection_filter(raw_results, num_qubits=3):
    """Apply manual postselection filtering and return statistics
    
    Args:
        raw_results: Dictionary of raw quantum state strings to counts
        num_qubits: Number of qubits (default 3)
        
    Returns:
        Dictionary containing:
        - 'filtered_results': Dictionary of filtered state strings to counts
        - 'total_events_before': Total events before filtering
        - 'total_events_after': Total events after filtering
        - 'rejected_events': Number of rejected events
        - 'rejected_states_count': Number of unique rejected states
        - 'accepted_states_count': Number of unique accepted states
        - 'rejection_rate': Rejection rate as percentage
    """
    from perceval.utils import BasicState
    
    total_events_before = sum(raw_results.values()) if raw_results else 0
    filtered_results = {}
    rejected_states = {}
    
    for state_str, count in raw_results.items():
        # Parse the state string to check postselection condition
        state = BasicState(state_str)
        # Check if exactly 1 photon in each qubit pair
        valid_state = True
        for i in range(0, 2*num_qubits, 2):
            if state[i] + state[i+1] != 1:
                valid_state = False
                break
        
        if valid_state:
            filtered_results[state_str] = count
        else:
            rejected_states[state_str] = count
    
    total_events_after = sum(filtered_results.values()) if filtered_results else 0
    rejected_events = sum(rejected_states.values()) if rejected_states else 0
    rejection_rate = (rejected_events / total_events_before * 100) if total_events_before > 0 else 0
    
    # Print statistics table
    print('M: postselection statistics:')
    print('┌─────────────────────────┬─────────────┐')
    print('│ Metric                  │ Value       │')
    print('├─────────────────────────┼─────────────┤')
    print('│ Events before postsel   │ %11d │' % total_events_before)
    print('│ Events after postsel    │ %11d │' % total_events_after)
    print('│ Rejected events         │ %11d │' % rejected_events)
    print('│ Rejected states count   │ %11d │' % len(rejected_states))
    print('│ Accepted states count   │ %11d │' % len(filtered_results))
    print('│ Rejection rate          │ %10.2f%% │' % rejection_rate)
    print('└─────────────────────────┴─────────────┘')
    
    return {
        'filtered_results': filtered_results,
        'total_events_before': total_events_before,
        'total_events_after': total_events_after,
        'rejected_events': rejected_events,
        'rejected_states_count': len(rejected_states),
        'accepted_states_count': len(filtered_results),
        'rejection_rate': rejection_rate
    }

def build_ghz_circuit(num_qubits=3, min_photons=None, processor_name="sim:belenos", is_ideal=False):
    """Build GHZ state generation circuit
    
    Args:
        num_qubits: Number of qubits (default 3)
        min_photons: Minimum photons filter (default num_qubits)
        processor_name: Remote processor name (default "sim:belenos")
        is_ideal: Use ideal local processor if True (default False)
        
    Returns:
        Configured processor ready for simulation
    """
    if min_photons is None:
        min_photons = num_qubits
        
    n_modes = 2 * num_qubits
    
    # Create processor
    if is_ideal:
        proc = pcvl.Processor("SLOS", n_modes)
    else:
        proc = pcvl.RemoteProcessor(processor_name, m=n_modes)

    # Get gate components
    cnotH = pcvl.catalog["heralded cnot"].build_processor()
    cnotP = pcvl.catalog['postprocessed cnot'].build_processor()
    
    # Apply minimum photon filter
    proc.min_detected_photons_filter(min_photons)
    
    # Build GHZ circuit: H + CNOTs
    proc.add(0, pcvl.BS.H())  # Hadamard on first qubit
    
    # Add CNOT gates for GHZ state
    for i in range(num_qubits - 1):
        if i == 0:
            proc.add(2*i, cnotH)  # First CNOT is heralded
        else:
            proc.add(2*i, cnotH)  # Additional CNOTs (could use cnotP for variety)
    
    # Set input state - one photon per qubit in mode 0 of each pair
    input_state_list = []
    for i in range(num_qubits):
        input_state_list.extend([1, 0])  # |0> state for each qubit
    
    input_state = pcvl.BasicState(input_state_list)
    proc.with_input(input_state)
    
    print('M: built GHZ circuit, qubits=%d, modes=%d, min_photons=%d' % (num_qubits, n_modes, min_photons))
    print('M: input_state=%s, processor=%s' % (str(input_state), proc.name))
    
    return proc


#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    isIdeal = not True; shots=20_000_000_000_000
    #isIdeal = True; shots=20_000
    
    # Build GHZ circuit using the function
    num_qubit = 3
    
    proc = build_ghz_circuit(num_qubits=num_qubit, processor_name="sim:belenos", is_ideal=isIdeal)
    pcvl.pdisplay(proc)
    
    sampler = Sampler(proc, max_shots_per_call=shots)
    
    if isIdeal:
        resD=sampler.sample_count(shots)
    else:
        # Run simulation once without postselection to get all raw data
        print('M: running simulation to collect all data...')
        job = sampler.sample_count.execute_async(shots)
        monitor_async_job(job)
        resD = job.get_results()
        
        # Apply postselection filtering using the function
        filter_stats = apply_postselection_filter(resD['results'], num_qubits=3)
        
        # Update results to show filtered data
        resD['results'] = filter_stats['filtered_results']

    #Handle different performance metrics based on simulator type
    if isIdeal:
        # Local/ideal simulator
        print('perf: global=%.6f' % resD['global_perf'])
    else:
        # Remote/noisy simulator
        print('perf: physical=%.6e, logical=%.6f' % (resD['physical_perf'], resD['logical_perf']))

    # Print results using the function and get bit-indexed dictionary
    bit_results = print_results(resD['results'])
    
    print('shots=%2g' % shots,proc.name)


