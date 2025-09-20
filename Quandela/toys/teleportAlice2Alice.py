#!/usr/bin/env python3
import argparse
import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler
from noisy_bell import monitor_async_job
from pprint import pprint


# ----------------- Alice's State Preparation -----------------

def build_alice_processor(p, theta=np.pi/4, phi=np.pi/3, verb=1):
    """Build Alice's processor: prepare state in modes 0-1"""
    
    # Prepare Alice's qubit: arbitrary superposition
    p.add(0, pcvl.BS(theta))
    p.add(1, pcvl.PS(phi))
    
    # Print Alice's secret state parameters
    if verb>1:
        print('[ALICE] Secret state parameters:')
        print('  theta=%.4f (%.2f degrees)' % (theta, theta*180/np.pi))
        print('  phi=%.4f (%.2f degrees)' % (phi, phi*180/np.pi))
    
    # Calculate and print Alice's secret state coefficients
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2) * np.exp(1j * phi)
    if verb>1:
        print('[ALICE] Secret state: |ψ⟩ = %.3f|0⟩ + (%.3f + %.3fi)|1⟩' % (alpha, beta.real, beta.imag))
        print('[ALICE] State probabilities: P(|0⟩)=%.3f, P(|1⟩)=%.3f' % (abs(alpha)**2, abs(beta)**2))
 
    return p


# ----------------- Alice's State Preparation and Measurement -----------------

def meas_basis( proc, basis='Z',  verb=1):
    """ measure it in specified basis"""
    # Add measurement basis rotation
    print(' Measurement basis: %s' % basis)
    if basis == 'X':
        #print('  Adding Hadamard for X-basis measurement')
        proc.add(0, pcvl.BS.H())  # H gate for X-basis
    elif basis == 'Y':
        #print('  Adding S†H for Y-basis measurement') 
        # For Y-basis: first S† (phase -π/2 on |1⟩), then H
        # In dual-rail: S† affects the relative phase between modes
        proc.add(1, pcvl.PS(-np.pi/2))  # S† gate: adds -π/2 phase to mode 1 (|1⟩)
        proc.add(0, pcvl.BS.H())        # H gate: mixes modes 0 and 1
    elif basis == 'Z':
        a=1 #print('  No rotation needed for Z-basis measurement')
    else:
        raise ValueError('Invalid basis: %s. Use X, Y, or Z' % basis)

def theory_prob1(theta, phi, basis):

    # Calculate theoretical probabilities from Alice's prepared state in specified basis
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2) * np.exp(1j * phi)
    
    # State in Z basis
    psi_z = np.array([alpha, beta], dtype=complex)  # state vector
    psi_z = psi_z / np.linalg.norm(psi_z)  # normalize

    # Basis transformation matrices
    U_z_to_x = (1/np.sqrt(2)) * np.array([[1, 1],
                                          [1, -1]], dtype=complex)
    U_z_to_y = (1/np.sqrt(2)) * np.array([[1,  1j],
                                          [1, -1j]], dtype=complex)

    if basis.upper() == 'Z':
        state=psi_z
    elif basis.upper() == 'Y':  # it may be messed-up, but this assignment give correct answers
        state=U_z_to_x @ psi_z
    elif basis.upper() == 'X':
        state=U_z_to_y @ psi_z
    else:
        raise ValueError("Basis must be 'Z', 'X', or 'Y'.")
    
    
    #Measures the probability of |1> in the Z basis.
    amplitude = state[1]
    return np.abs(amplitude)**2
    

#########################
#  MAIN
#########################

# ----------------- Run Simulation -----------------

def run_alice_measurement(backend, shots,init_state, basis='Z',verb=1):
   
    num_qubit=1
    nMode=2*num_qubit
    minPhoton=num_qubit

    # Parse initial state parameters
    theta, phi = init_state[0], init_state[1]
    
    # Simple 2-mode processor for Alice's qubit
    if backend=='SLOS':
        proc = pcvl.Processor("SLOS", nMode)
    if backend=='ascella':
        proc = pcvl.RemoteProcessor("sim:ascella",m=nMode)
    if backend=='belenos':
        proc = pcvl.RemoteProcessor("sim:belenos",m=nMode)

    print('[INFO] Shots: %d, Min photons: %d, Basis: %s, backend: %s' % (shots, minPhoton, basis,proc.name))
   
 
    # Build Alice's processor
    proc= build_alice_processor(proc,  theta, phi,verb)
    meas_basis( proc,basis,verb)
    
    # Input state: 1 photon in mode 0 (Alice starts with |0⟩)
    input_state = pcvl.BasicState([1, 0])
    proc.with_input(input_state)
    proc.min_detected_photons_filter(minPhoton)
    
    # Display Alice's circuit
    print('\n[CIRCUIT]:')
    pcvl.pdisplay(proc, recursive=False)

    if verb>1: 
        print('\n[MEASUREMENT] Running measurement...')
    sampler = Sampler(proc, max_shots_per_call=shots)
    if backend=='SLOS':
        res = sampler.sample_count(shots)
    else:
        job = sampler.sample_count.execute_async(shots)
        monitor_async_job(job)
        res=job.get_results()
    
    # Step 1: Print Alice's measurements (modes 0-1)
    print("\n[STEP 1] Alice's measurements (modes 0-1):")
    total_counts = sum(res["results"].values())
    print("  Total measured events: %d" % total_counts)
    pprint(res["results"])
    alice_raw_results = { state_str: count for state_str, count in res["results"].items()}
    

    # Step 2: Filter out invalid outcomes
    print("\n[STEP 2] Filtering invalid outcomes...",proc.name)
    valid_results = {}
    invalid_results = {}
    
    for state_str, count in res["results"].items():
        state = pcvl.BasicState(state_str)
        # Valid measurement: exactly 1 photon in Alice's modes (0,1)
        if state[0] + state[1] == 1:  
            valid_results[state_str] = count
        else:
            invalid_results[state_str] = count
    
    valid_counts = sum(valid_results.values())
    invalid_counts = sum(invalid_results.values())
    
    # Step 3: Print rejection statistics
    print("\n[STEP 3] Rejection statistics:")
    print("  Total states measured: %d" % len(res["results"]))
    print("  Valid state patterns: %d" % len(valid_results))
    print("  Invalid state patterns: %d" % len(invalid_results))
    print("  Valid events: %d/%d (%.2f%%)" % (valid_counts, total_counts, valid_counts/total_counts*100 if total_counts > 0 else 0))
    print("  Rejected events: %d/%d (%.2f%%)" % (invalid_counts, total_counts, invalid_counts/total_counts*100 if total_counts > 0 else 0))
    print("  Measurement success rate: %.2f%%" % (valid_counts/total_counts*100 if total_counts > 0 else 0))

    
    print("\n[STEP 4] Valid Alice measurements only:")
    if not valid_results:
        print("  No valid outcomes found!")
        exit(99)
    pprint(valid_results)
        
    # Compare Alice's measurement with Alice's true state
    print('\n[COMPARISON] true state vs  measured state, shots=%.2e, 1/sqrt(shots)=%.3f'%(shots, 1/np.sqrt(shots)))
    
    p1=theory_prob1( theta, phi, basis)
    p_1_theory=p1; p_0_theory=1.-p1
    
    
    # Extract Alice's measured probabilities
    p_0_alice = valid_results.get(pcvl.BasicState([1, 0]), 0.0)/valid_counts
    p_1_alice = valid_results.get(pcvl.BasicState([0, 1]), 0.0)/valid_counts
    
    print('  P(|0⟩)  true= %.3f   meas=%.3f   diff=%.3f' % (p_0_theory,p_0_alice, p_0_theory - p_0_alice))
    print('  P(|1⟩)  true= %.3f   meas=%.3f   diff=%.3f' % (p_1_theory,p_1_alice, p_1_theory - p_1_alice))
        
    # Calculate fidelity metric
    fidelity = np.sqrt(p_0_theory * p_0_alice) + np.sqrt(p_1_theory * p_1_alice)
    print('  State preparation fidelity: %.4f' % fidelity)
    
    if fidelity > 0.99:
        print('  ✓ Excellent state preparation!')
    elif fidelity > 0.95:
        print('  ○ Good state preparation')
    else:
        print('  ✗ Poor state preparation quality')


# ----------------- CLI Entry -----------------

def main():
    parser = argparse.ArgumentParser(description="Alice State Preparation and Measurement in Perceval 1.0")
    parser.add_argument("-b","--backend", type=str, default="SLOS", help="Backend to use: 'SLOS' or 'ascella' or 'belenos'")
    parser.add_argument("-n","--shots", type=int, default=10_000, help="Number of shots for finite-shot simulation")
    parser.add_argument('-B',"--basis", type=str, default="Z", choices=['X', 'Y', 'Z'], help="Measurement basis: X, Y, or Z")
    parser.add_argument('-i',"--initState", type=float, nargs=2, metavar=('THETA', 'PHI'), default=[0.9,-2.2],
                       help="Initial state parameters: theta (BS angle) and phi (PS phase) in radians")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level: 0=minimal, 1=normal, 2=detailed")

    args = parser.parse_args()
    print(vars(args))
    
    run_alice_measurement(args.backend, args.shots, args.initState, args.basis,args.verb)

if __name__ == "__main__":
    main()


    # [ALICE] Secret state: |ψ⟩ = 0.707|0⟩ + (0.707 + 0.000i)|1⟩
# 481  ./bb2.py -n 10_000_000 -i 0.9 2.2
#  482  ./bb2.py -n 10_000_000 -i 2.3 -4.2
 
