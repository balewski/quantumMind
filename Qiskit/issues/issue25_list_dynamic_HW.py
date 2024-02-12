
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()

backL=service.backends(simulator=False, operational=True)
print('seeA  operational HW', len(backL), backL )
for backend in backL:
    config = backend.configuration()
    attrL=  getattr(config, "supported_features", [])
    print('sss',backend,attrL)
    if  "qasm3" in  attrL : print('support QASM3')
  

backL=service.backends(dynamic_circuits=True)
print('see2 dynamic_circ', backL )


#In IBM Quantum systems list, systems that support running dynamic circuits have the label "OpenQASM 3":
#  "qasm3" not in getattr(self.configuration(), "supported_features", []):
