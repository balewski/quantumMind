#!/usr/bin/env python3
'''  run on Fake backend w/ Qubic 1.0

From Siddharth

 I used Qiskit 1.0.1 and Qiskit Runtime Service 0.20 for this program. 

'''
import qiskit as qk
from qiskit.visualization import circuit_drawer
import site
import os
import json
from datetime import datetime
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import fake_backend

from pprint import pprint


#...!...!....................
class BackendEncoder(json.JSONEncoder):
    """A json encoder for qobj"""

    def default(self, o):
        # Convert numpy arrays:
        if hasattr(o, "tolist"):
            return o.tolist()
        # Use Qobj complex json format:
        if isinstance(o, complex):
            return [o.real, o.imag]
        if isinstance(o, ParameterExpression):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

#...!...!....................
class FakeBackend(fake_backend.FakeBackendV2):
    
    def __init__(self, backend):
        package_name = 'qiskit_ibm_runtime'
        package_dirs = site.getsitepackages()
        
        backends_dir = None
        for dir in package_dirs:
            package_path = os.path.join(dir, package_name)
            if os.path.exists(package_path):
                backends_dir = os.path.join(package_path, "fake_provider", "backends")
                break
        #print('bb1',backends_dir)
        backends_dir = 'my_fake_provider/'
        backend_og_name = backend.name.split("_")[1]
        self.conf_filename = f"conf_{backend_og_name}.json" 
        self.props_filename = f"props_{backend_og_name}.json" 
        self.defs_filename = f"defs_{backend_og_name}.json"
        self.backend_name = f"fake_{backend_og_name}"
        #print('pp,',backends_dir, backend_og_name)
        self.dirname = os.path.join(backends_dir, backend_og_name)
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)
            config = backend.configuration()
            props = backend.properties()
            defs = backend.defaults()
            if config:
                config_path = os.path.join(self.dirname, self.conf_filename)
                config_dict = config.to_dict()
                with open(config_path, "w") as fd:
                    fd.write(json.dumps(config_dict, cls=BackendEncoder))
            if props:
                props_path = os.path.join(self.dirname, self.props_filename)
                with open(props_path, "w") as fd:
                    fd.write(json.dumps(props.to_dict(), cls=BackendEncoder))
            if defs:
                defs_path = os.path.join(self.dirname, self.defs_filename)
                with open(defs_path, "w") as fd:
                    fd.write(json.dumps(defs.to_dict(), cls=BackendEncoder))

        super().__init__()

#...!...!....................
def create_ghz_circuit(n):
    qc = qk.QuantumCircuit(n)
    with qc.if_test((qc.cregs[0], 1)): qc.x(1)  # NEW
    for i in range(1, n):  qc.cx(0, i)    
    qc.measure(1, 0)

    return qc

#...!...!....................
def feedForw_circuit():
    qc = qk.QuantumCircuit(2,1,name='ansatz_circ')
    for i in range(2): qc.rx(0.65,i)
    qc.cx(1,0)
    qc.measure(1, 0)
    with qc.if_test((qc.cregs[0], 1)): qc.x(1) 
    return qc

#=================================
if __name__ == "__main__":
    
    service = QiskitRuntimeService()
    instance = service.instances()[0] # Or put your instance manually
    print(instance)
    backName="ibm_torino"
    
    true_backend = service.get_backend(backName)
    fake_backend = FakeBackend(true_backend)
    
    print('M: constructed   backend:',fake_backend.name)
    #qc=create_ghz_circuit(5)
    qc=feedForw_circuit()
    print(qc)

    qcT = qk.transpile(qc, backend=fake_backend, optimization_level=3, seed_transpiler=44)
    print('circuit transpiled for fake: ', backName)
    print(circuit_drawer(qcT, output='text',cregbundle=True,idle_wires=False))

    print('transpiled CX-depth:',qcT.depth(filter_function=lambda x: x.operation.num_qubits == 2 ))
    
    print('job started locally ,  nq=%d  at %s ...'%(qcT.num_qubits,fake_backend.name))
    job =  fake_backend.run(qcT,shots=1000)

    result=job.result()
    counts=result.get_counts()
    print('M:counts:')
    pprint(counts)

    print('M:ok')

