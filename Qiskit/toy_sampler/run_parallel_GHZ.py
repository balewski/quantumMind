#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
spreads a GHZ small circuit in many copies on a QPU based on user's choice of qubits

'''
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from qiskit.result.utils import marginal_distribution

# https://qiskit.org/documentation/stubs/qiskit.visualization.circuit_drawer.html
# https://qiskit.org/ecosystem/ibm-runtime/stubs/qiskit_ibm_runtime.options.Options.html#qiskit_ibm_runtime.options.Options
    
from qiskit.tools.visualization import circuit_drawer
from time import time
from pprint import pprint
import numpy as np
from qiskit.tools.monitor import job_monitor
from collections import Counter


'''
Ask:
options.optimization_level ? 
default is 3, which is heavy optimization - if I give  the layout to transpiler do I want any optimization to happen?

quasi_probs are sometimes negative - do I need to clip them?
can I get stat eerror of quasiprob


from qiskit.tools.monitor import job_monitor

  File "/usr/local/lib/python3.10/dist-packages/qiskit/tools/monitor/job_monitor.py", line 105, in job_monitor
    _text_checker(
  File "/usr/local/lib/python3.10/dist-packages/qiskit/tools/monitor/job_monitor.py", line 49, in _text_checker
    msg += " (%s)" % job.queue_position()
AttributeError: 'RuntimeJob' object has no attribute 'queue_position'

'''

import argparse
def commandline_parser():  # used when runing from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3], help="increase output verbosity", default=1, dest='verb') 

    parser.add_argument('-n','--num_shot',type=int,default=4000, help="shots")
    parser.add_argument('-b','--backend',default="ibmq_qasm_simulator", help="tasks")    
    parser.add_argument('--spam_corr',type=int, default=1, help="Mitigate error associated with readout errors, 0=off")
    parser.add_argument( "-F","--fakeSimu", action='store_true', default=False, help="will switch to backend-matched simulator")
    parser.add_argument( "-M","--useQMap", action='store_true', default=False, help="use provided qubit map")

    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if 1:
        assert args.backend in ["ibmq_qasm_simulator",
                        "ibmq_kolkata","ibmq_mumbai","ibm_algiers","ibm_hanoi", "ibm_cairo",# 27 qubits
                        "ibm_brisbane", 'ibm_nazca' # 127 qubits
                                 ]
    return args

#...!...!....................
def create_ghz_circuit(n):
    qc = QuantumCircuit(n,n-1)
    qc.h(0)
    for i in range(1, n):  qc.cx(0, i)
    qc.barrier()
    for i in range(1, n):  qc.measure(i,i-1)
    return qc

#...!...!....................
def parallel_placement_and_transpile(small_circ, layout, backend):
    reps = len(layout)
    num_qubits = small_circ.num_qubits
    num_clbits = small_circ.num_clbits

    nqTot=num_qubits*reps
    nbTot=num_clbits*reps
    #print('PPAT: nqTot=%d'%(nqTot))
    circuit = QuantumCircuit(nqTot, nbTot)
    bits_layout=[]
    for i in range(reps):
        q0=i*num_qubits
        b0=i*num_clbits
        print('q0,b0',q0,b0,' physQ:',layout[i])
        bitsL=[i for i in range(b0,b0+num_clbits)]
        circuit.compose(small_circ, range(q0,q0+num_qubits),bitsL, inplace=True)
        bits_layout.append(bitsL)
        
    #.... flatten  qubits layout
    if layout[0]==None:
        flat_list=None
        print('transpile map: None')
    else:   
        flat_list = [item for sublist in layout for item in sublist]
    T0=time()

    trans_circuit = transpile(circuit, backend  , initial_layout=flat_list)
    elaT=time()-T0
    print('transpile on %d qubits took elaT=%.1f sec'%(nqTot,elaT))
    return trans_circuit,reps,bits_layout

#...!...!....................
def meas_int2bits(probsI,nclbit): # converts int(mbits) back to bitstring
    print('MI2B: repack %d keys'%(len(probsI)))
    probs = {}
    for key, val in probsI.items():
        mbit=format(key,'0'+str(nclbit)+'b')
        probs[mbit] = val
        #print(key,mbit,val)
    return probs

#...!...!....................
def useQMap(backN):
    
    if backN=='ibmq_qasm_simulator':
        multiQMap=[ [0,1,2,3], [5,8,7,6] ]
    if backN=='ibm_hanoi' or  args.backend=='ibm_cairo' :
        multiQMap= [[12, 10, 13, 15], [1, 0, 4, 2]]
    if backN=='ibm_brisbane':
        multiQMap= [[41, 40, 42, 53], [66, 65, 73, 67], [81, 72, 82, 80], [96, 109, 95, 97], [104, 111, 105, 103], [77, 76, 78, 71], [22, 23, 21, 15], [26, 16, 25, 27], [30, 17, 29, 31], [118, 110, 119, 117]]
    if backN=='ibm_nazca':  #8 locations
        multiQMap= [[98, 97, 91, 99], [102, 92, 101, 103], [114, 113, 109, 115], [4, 3, 15, 5], [20, 33, 19, 21], [41, 40, 42, 53], [106, 105, 93, 107], [122, 121, 123, 111]]

    return multiQMap
        

#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=commandline_parser()
    np.set_printoptions(precision=3)

    if args.useQMap:
        multiQMap=useQMap(args.backend)
    else:
        multiQMap=[None,None]
        
    print('M: acquire backend:',args.backend)
    service = QiskitRuntimeService()
    backend = service.get_backend(args.backend)
    print('use backend version:',backend.version )
 
    options = Options()
    options.optimization_level=3 #  even heavier optimization (deafult=3)
    options.resilience_level = args.spam_corr  # Mitigate error associated with readout errors, (default=1, 0=off)
    options.transpilation.skip_transpilation = True

    if args.fakeSimu:  # use fake_hanoi noise model
        from qiskit_aer.noise import NoiseModel
        noisy_backend = service.get_backend('ibm_hanoi')
        backend_noise_model = NoiseModel.from_backend(noisy_backend)
        options.simulator = {  "noise_model": backend_noise_model }
        print('add noisy_backend FAKE =',noisy_backend )
        #multiQMap= [[12, 10, 13, 15], [1, 0, 4, 2]]
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)

    qcP=create_ghz_circuit(4)
    print('M: one circ:');print(qcP)

    qcT,nReps,bitsLL = parallel_placement_and_transpile(qcP, multiQMap, backend)
    if args.verb>1: print(circuit_drawer(qcT.decompose(), output='text',cregbundle=False))
    if args.verb>1: print('M:bitsLL:',bitsLL)

    print('M:run circuit ... :',backend)
    T0=time()
    job = sampler.run(qcT,shots=args.num_shot)
    #1job_monitor(job, interval=5, quiet=False)
    elaT=time()-T0
    ic=0
    probsI=job.result().quasi_dists[ic] # just 1 circuit
    print('M: job ended elaT=%.1f sec'%(elaT))    

    if args.verb>1: print('M:probs:%s'%(probsI))

    probsB=meas_int2bits(probsI,qcT.num_clbits)
    
    if args.verb>1: print('probsB:'); pprint(probsB)
    sumC=Counter()
    for ir in range(nReps):
        probs=marginal_distribution(probsB, indices=bitsLL[ir])
        print('\ncirc copy=%d, qL:%s probs:'%(ir,multiQMap[ir]))
        probsC=Counter(probs)
        sumC+=probsC
        # Sorting the Counter by value
        sortedT = sorted(probsC.items(), key=lambda item: item[1], reverse=True)
        #print(sortedT)
        okProb=0
        for key,val in sortedT:
            if key in ['000','111']: okProb+=val
            if ir <4: print('%s  %8.4f'%(key,val))                
        print('   000+111 prob=%.3f'%okProb)

    # Dividing each count by nReps
    for key in sumC: sumC[key] /= nReps
    sortedT = sorted(sumC.items(), key=lambda item: item[1], reverse=True)
    print('\nsumC:')
    for key,val in sortedT:   print('%s  %8.4f'%(key,val))    
    print('M: done, shots:',args.num_shot,backend)
    print(qcP)


    

'''
./run_parallel_GHZ.py -b ibm_nazca -n 5000 -v2

problem: 3 bits*14=42 --> 
MI2B: repack 4998 keys
probsB:
{'000000000000000000000101111011111000000110': 0.0002001305090559442,
 '000000000000000000000111000000100000000000': 0.00019999973016919745,
 '000000000000000000110111000011010000000010': 0.00019999972975218437,
...


circ copy=0, probs:
011    0.1310
101    0.1726
001    0.1336
010    0.1216
000    0.1112
111    0.1390
110    0.0906
100    0.1004



circ copy=13, probs:
011    0.0064
101    0.0192
001    0.0286
010    0.0190
000    0.4552
111    0.4336
110    0.0284
100    0.0096
M: done, shots: 5000 <IBMBackend('ibm_nazca')>

'''
