#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Bell state generation with custom density matrix and noise modeling (needs cleanup)

Example of defining custom delays with different pauli errors
Use sampler 

Most of code written by ChatGPT & Preplexity
May need some cleanup to work
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel,  depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit.circuit import Gate
from qiskit.circuit.library import IGate

# custom transpiler
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel


import argparse
#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument('-q','--numQubits', default=3, type=int,  help='num qubits') 
    # Qiskit:
    parser.add_argument('-n','--numShots', default=800, type=int, help='num of shots')
  
    args = parser.parse_args()

    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
    return args


# Define custom identity gates to mimic delays
class GateMoveA(Gate):
    def __init__(self):
        super().__init__("move_a", 1, [])
    def _define(self):
        qc = QuantumCircuit(1)
        qc.id(0)
        self.definition = qc


#...!...!....................
def ghzCirc(nq=2):
    """Create a GHZ state preparation circuit."""
    qc = QuantumCircuit(nq, nq)
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    qc.ry(np.pi/3,1)
    qc.append(GateMoveA(), [0])
    qc.barrier()
    qc.measure(range(nq), range(nq))
    return qc


#...!...!....................
def inspectCirc(qc):
    print('M:list circuit instructions')
    for inst, qargs, cargs in qc.data:
        print(inst,qargs, cargs)

#...!...!..................
def my_noise_model(spam=True,therm=True,depol=True):
    # Define parameters
  
    t1 = 50e-6  # 50 microseconds
    t2 = 70e-6  # 70 microseconds
    if not therm:
        t1=999; t2=999
        
    p_depol1 = 0.001  # Single-qubit depolarizing error probability
    p_depol2 = 0.01   # Two-qubit depolarizing error probability
    if not depol:
        p_depol1 =1e-9; p_depol2 =1e-9
        
    p_set0_meas1 = 0.01  # Probability of measuring 1 when qubit is in state 0
    p_set1_meas0 = 0.02  # Probability of measuring 0 when qubit is in state 1

    # Define gate durations (in seconds)
    # 'p': 0,   # Virtual gate, no duration
    gate_times = {
        'sx': 25e-9,  # 25 nanoseconds for sqrt(X) gate
        'cz': 300e-9  # 300 nanoseconds for CZ gate
    }

    # Create noise model
    noise_model = NoiseModel()

    # Add errors to all gates
    for gate, time in gate_times.items():
        if gate in ['sx']:
            # Single-qubit gate error
            thermal_error = thermal_relaxation_error(t1, t2, time)
            depol_error = depolarizing_error(p_depol1, 1)
            combined_error = depol_error.compose(thermal_error)
            noise_model.add_all_qubit_quantum_error(combined_error, gate)
        elif gate == 'cz':
            # Two-qubit gate error
            thermal_error = thermal_relaxation_error(t1, t2, time).tensor(thermal_relaxation_error(t1, t2, time))
            depol_error = depolarizing_error(p_depol2, 2)
            combined_error = depol_error.compose(thermal_error)
            noise_model.add_all_qubit_quantum_error(combined_error, gate)
          
    # Add asymmetric readout error to all measurements
    readout_error = ReadoutError([[1 - p_set0_meas1, p_set0_meas1], 
                              [p_set1_meas0, 1 - p_set1_meas0]])
    if spam: noise_model.add_all_qubit_readout_error(readout_error)
    #1print(noise_model)
    return  noise_model

#...!...!..................
def print_noise_model(noise_model):
    print("\nNoise Model Details:")
    print(f"Basis gates: {noise_model.basis_gates}")

    print("\nInstructions with noise:")
    noiseD=noise_model.to_dict()['errors']

    for g in noiseD:
        gateN=g['operations'][0]
        #print(g)
        pV=np.array(g['probabilities'])
        print(gateN,pV.shape,'pauliErr:',pV,'sum:',sum(pV))
       
    print()

#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    # Set NumPy print options to display full matrices with 3 decimal places
    np.set_printoptions(threshold=1e-5, linewidth=np.inf, precision=3)

    args=get_parser() 

    qc=ghzCirc(args.numQubits)
    inspectCirc(qc)
    print(qc)


    # construct noise model
    noise_model=my_noise_model(spam=False)#,therm= False,depol= False)
    print_noise_model(noise_model)
  
    print("Noise model constructed.")

    # Set up the simulator and sampler
    backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        basis_gates=[ "ry", "u3", "cx", "id"]
    )
    print("Using backend =", backend.name)

    basis_gates = ['ry', 'u3', 'cx', 'id','move_a']
    pm = PassManager()
    pm.append(BasisTranslator(sel, basis_gates))
    qcT = pm.run(qc)
    print(qcT)

    inspectCirc(qcT)
    
    options = SamplerOptions()
    options.default_shots=args.numShots
    
    sampler = Sampler(mode=backend, options=options)

    
    job = sampler.run((qcT,))
    jobRes=job.result()
    
