from qiskit_ibm_runtime import QiskitRuntimeService
backend_name = 'ibm_hanoi'  # Replace with your actual backend name
backend_name='ibm_brisbane'


service = QiskitRuntimeService()
backend = service.backend(backend_name)
print('got backend:',backend)
properties = backend.properties()
print('got properties for ',backend)

gateN='ecr'

for gate in properties.gates:
    if gate.gate == gateN:
        q1, q2 = gate.qubits
        fidelity = 1-properties.gate_error(gateN, [q1, q2])
        print('cx:',q1,q2,fidelity)
    else:
        print('gate:',gate.gate)
