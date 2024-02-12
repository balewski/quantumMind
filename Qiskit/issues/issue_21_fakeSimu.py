#!/usr/bin/env python3
'''  compile for fake simulator

From aziz
Hi Jan,
 
I confirm, you can ignore this warning when you use skip_transpilation = True.
'Optimization level clipped from 3 to 1'}

 You can also delete the line options.optimization_level=3 since it will not be performed.

'''
import qiskit as qk
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from pprint import pprint
import argparse

#...!...!....................
def create_ghz_circuit(n):
    qc = qk.QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc


#=================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="QPU running job")
    parser.add_argument( "-F","--fakeSimu", action='store_true', default=False, help="will switch to backend-matched simulator")
    args = parser.parse_args()
    
    qc=create_ghz_circuit(5)
    print(qc)

    service = QiskitRuntimeService()
    options = Options()
    options.transpilation.skip_transpilation = True  # I will transpile explicitely here (default is False)
    options.execution.shots=1000
    options.resilience_level = 1  # Mitigate error associated with readout errors, (default=1, 0=off)

    if args.fakeSimu:  # use fake_hanoi noise model
        from qiskit_aer.noise import NoiseModel
        noisy_backend = service.get_backend(args.backend)
        options.simulator = {
            "noise_model":  NoiseModel.from_backend(noisy_backend),
            "basis_gates": noisy_backend.configuration().basis_gates,
            "coupling_map": noisy_backend.configuration().coupling_map,
        }
        qcT = qk.transpile(qc, backend=noisy_backend, optimization_level=3)
        print('add noisy_backend =',noisy_backend.name )
        backend = service.get_backend("ibmq_qasm_simulator")
    else:
        print('M: acquire backend:',args.backend)
        backend = service.get_backend(args.backend)
        qcT = qk.transpile(qc, backend=backend, optimization_level=3, seed_transpiler=44)
     
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)

    #... manual transpiler
   
    print('transpiled CX-depth:',qcT.depth(filter_function=lambda x: x.operation.num_qubits == 2 ))
  
    print(qcT.draw(output='text',idle_wires=False))  # skip ancilla
    
    print('job started,  nq=%d  at %s ...'%(qcT.num_qubits,backend.name))
    job = sampler.run(qcT)
    result=job.result()
    jobMD=result.metadata    
    print('M: job-meta[0]:',jobMD[0])

    print('M:qprobs:')
    pprint(result.quasi_dists[0])

    print('M:ok')

