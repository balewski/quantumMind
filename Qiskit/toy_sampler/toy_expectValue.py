from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

#...!...!....................
def create_h_circuit(n):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.x(i)
    qc.barrier()
    for i in range(0,n):  qc.measure(i,i)
    return qc

nq=3
qc=create_h_circuit(nq)
print(qc)

#obs = SparsePauliOp("I" * nq)  # ev=1
obs = SparsePauliOp("IIZ") #  ev=0
obs = SparsePauliOp("IZI") #  ev=-1.0
obs = SparsePauliOp("ZZI") #  ev=1.0
obs = SparsePauliOp("IZZ") #  ev=0.0 
print('obs:',obs)

backend = AerSimulator()
print('job started,  nq=%d  at %s ...'%(qc.num_qubits,backend.name))
options = SamplerOptions()
options.default_shots=1000
estimator = Estimator(backend) #, options=options)

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_circuit = pm.run(qc)
isa_observable = obs.apply_layout(isa_circuit.layout)
 
job = estimator.run([(isa_circuit, isa_observable)])
result = job.result()
rdata=result[0].data
print("Expectation value: %.3f +/- %.3f "%(rdata.evs,rdata.stds))
print("Metadata: ",result[0].metadata)

