#!/usr/bin/env python3
''' problem: ??

'''
import qiskit as qk
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from pprint import pprint
import argparse
from mthree._helpers import system_info as backend_info
from collections import Counter

#...!...!....................
def create_ghz_circuit(n):
    qc = qk.QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.measure_all()
    return qc


#...!...!....................
def circ_depth(qc,text='myCirc'):   # from Aziz @ IBMQ    summer 2023
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    opsD=qc.count_ops()
    #pprint(opsD)
    
    if 'cx' in opsD: d2q= opsD['cx']
    elif 'cz' in opsD: d2q= opsD['cz']
    else: d2q= opsD['ecr']
    print('%s-circ 2q depth: %d, num 2q: %d'%(text,len2,d2q))


#=================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="QPU running job")
    parser.add_argument( "-F","--fakeSimu", action='store_true', default=False, help="will switch to backend-matched simulator")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time, test before use ")

    parser.add_argument('--read_mit',type=int, default=1, help="Mitigate readout errors, 0=off")
    args = parser.parse_args()
    
    qc=create_ghz_circuit(6)
    print(qc)

    #.... construct sampler-job w/o backend...
    print('M: activate QiskitRuntimeService() ...')
    service = QiskitRuntimeService()    
    options = Options()
    options.transpilation.skip_transpilation = True  # I will transpile explicitely here (default is False)
    options.execution.shots=4000
    options.optimization_level=3  #  even heavier optimization (deafult=3)
    options.resilience_level = args.read_mit  # Mitigate  readout errors, (default=1, 0=off)

    #... manual transpiler  for the backend you have chosen
    if args.fakeSimu:  # use  noise model
        from qiskit_aer.noise import NoiseModel
        noisy_backend = service.get_backend(args.backend)
        backend_noise_model = NoiseModel.from_backend(noisy_backend)
        options.simulator = {  "noise_model": backend_noise_model }
        print('use noisy_backend FAKE =',noisy_backend )
        pprint(backend_info(noisy_backend ))
        qcT = qk.transpile(qc, backend=noisy_backend, optimization_level=3, seed_transpiler=42)
        backend = service.get_backend("ibmq_qasm_simulator")
    else:
        backend = service.get_backend(args.backend)
        qcT = qk.transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
        

    print(qcT.draw(output='text',idle_wires=False))  # skip ancilla
    circ_depth(qc,'ideal')
    circ_depth(qcT,'transpiled')

    print('M: run on backend:',backend.name)
    
    #...  attach backend to sampler
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)
    
    if not args.executeCircuit:
        print('NO execution of circuit, use -E to execute the job')
        exit(0)

    print('job started,  nq=%d  at %s ...'%(qcT.num_qubits,backend.name))
    job = sampler.run(qcT)
    result=job.result()    
    jobMD=result.metadata    
    print('M: job-meta[0]:',jobMD[0])

    quasis=result.quasi_dists
    #1pprint(quasis)
    quasisCL=[ Counter(data) for data in quasis]

    nTop=5; ic=0
    print('\nM: top %d quasis of %d :'%(nTop,len(quasisCL[ic])))
    print(quasisCL[ic].most_common(nTop))
    

    print('M:ok')

