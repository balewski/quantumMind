#!/usr/bin/env python3
'''
apply ZNE based on 
https://cqcl.github.io/Qermit/zero_noise_extrapolation.html

Use Qiskit Aer
it plots, but only works on PM

'''

#...!...!....................
def circ_depth_aziz(qc,text='myCirc'):   # from Aziz @ IBMQ
    len1=qc.depth(filter_function=lambda x: x.operation.num_qubits == 1)
    len2x=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    len3=qc.depth(filter_function=lambda x: x.operation.num_qubits ==3 )
    len4=qc.depth(filter_function=lambda x: x.operation.num_qubits > 3 )
    nq=qc.num_qubits
    outD={'cx':len2x,'2q':len2,'3q':len3,'4q+':len4,'nq':nq}
    print('%s depth:'%text,outD)
    return [len1,len2,len3,len4]

#...!...!....................
def myMethods(my_object):
        methods = [method for method in dir(my_object) if callable(getattr(my_object, method))]
        class_name = my_object.__class__.__name__
        print(class_name)
        # Print the list of methods
        print(methods)

print('M:start...')

from qermit import ObservableExperiment, AnsatzCircuit, SymbolsDict, ObservableTracker
from pytket.utils import QubitPauliOperator
from pytket.pauli import QubitPauliString
from pytket.pauli import Pauli
from pytket import Circuit, OpType
from pytket.circuit.display import render_circuit_jupyter
from qermit.zero_noise_extrapolation import gen_ZNE_MitEx, Folding, Fit
from pytket.extensions.qiskit import IBMQEmulatorBackend
# jan
from pytket.extensions.qiskit import tk_to_qiskit
import qiskit as qk
from pprint import pprint

tk_backend = IBMQEmulatorBackend(
    backend_name='ibmq_jakarta',
    instance='',
)

noise_scaling_list = [1.5, 2, 2.5, 3, 3.5]
zne_mitex = gen_ZNE_MitEx(
    backend=tk_backend,
    noise_scaling_list=noise_scaling_list,
    show_fit=True,
    folding_type=Folding.two_qubit_gate,
    fit_type=Fit.exponential
)
zne_mitex.get_task_graph()

n_qubits = 5 ; n_controls = 3
#n_qubits = 4 ; n_controls = 2
n_layers=1
shots=10000

circuit = Circuit(n_qubits=n_qubits)
for _ in range(n_layers):
    for i in range(n_qubits-n_controls):
        circuit.add_gate(OpType.CnX, [i+j for j in range(n_controls+1)])
    for i in reversed(range(n_qubits-n_controls)):
        circuit.add_gate(OpType.CnX, [i+j for j in range(n_controls+1)])

    
#render_circuit_jupyter(circuit)
# run the circuit on noisy simulator using Qiskit
qc0=qk.QuantumCircuit(n_qubits,n_qubits-2)
qc1=qc0.compose(tk_to_qiskit(circuit))
for i in range(n_qubits-2): qc1.measure(i+2,i) # measure  last few  qubits
print(qc1)
circ_depth_aziz(qc1)
if 1:
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.fake_provider import FakeJakarta
    from qiskit.tools.monitor import job_monitor
    qk_backend = AerSimulator.from_backend(FakeJakarta())
    qcT = qk.transpile(qc1, backend=qk_backend,  optimization_level=3)
    circ_depth_aziz(qcT,text='inp_trans qiskit '+str(qk_backend))    
    #1print(qcT)
    job =  qk_backend.run(qcT,shots=shots)
    jid=job.job_id()
    print('submitted JID=',jid ,' now wait for execution of your circuit ...')    
    counts = job.result().get_counts(qcT)
    print('counts:',counts)
    
        
ansatz = AnsatzCircuit(
    Circuit=circuit,
    Shots=shots,
    SymbolsDict=SymbolsDict()
)
#1myMethods(ansatz)
'''
AnsatzCircuit
['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__dir__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_asdict', '_make', '_replace', 'count', 'index']
'''

qps = QubitPauliString(
    qubits=circuit.qubits,
    paulis=[Pauli.Z for _ in circuit.qubits]
)

qpo = QubitPauliOperator({qps:1})
obs = ObservableTracker(qubit_pauli_operator=qpo)
obs_exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs)

outL=zne_mitex.run(
    mitex_wires=[obs_exp]
)
# aa is a list
print('M: size=',len(outL))
myMethods(outL[0])

'''
QubitPauliOperator
['__add__', '__class__', '__delattr__', '__dir__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__le__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmul__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '_collect_qubits', 'compress', 'dot_state', 'from_list', 'get', 'state_expectation', 'subs', 'to_list', 'to_sparse_matrix']
'''


exit(0)
from pytket.extensions.qiskit import IBMQEmulatorBackend
from qiskit import Aer, QuantumCircuit
from pprint import pprint


shots=1000
'''
uc_spam_mitres = gen_UnCorrelated_SPAM_MitRes(
   backend = lagos_backend,
   calibration_shots = 500
)
uc_spam_mitres.get_task_graph()
'''

nq=4
qc=QuantumCircuit(nq, nq)
ctrL=[i for i in range(nq-1)]
qc.mcx(ctrL,nq-1)
for i in range(nq): qc.measure(i,i)
print(qc)
job=backend.run(qc,shots=shots)
counts = job.result().get_counts()
print('counts:',backend); pprint(counts)
    

