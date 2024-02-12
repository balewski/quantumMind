#!/usr/bin/env python3
'''  access methods for GroverResult 
https://qiskit.org/documentation/stable/0.26/stubs/qiskit.aqua.algorithms.GroverResult.html
are not accessible

Answer:
The documentation you linked is for an older GroverResult object from Qiskit Aqua. Aqua has been deprecated, and has had most of its modules migrated into Qiskit Terra under the algorithms module. The documentation for Terra's GroverResult object is here.
 
There is no way to retrieve the circuit directly from the GroverResult object, however you can retrieve it through the Grover object. The Grover object has a construct_circuit method. The result object has iterations,  which is a list of powers at each iteration. The last element was where it ended and returned the result. So given the problem you ran, you can use that along with the result to get the circuit. 


Follow tutorial:
https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/algorithms/06_grover.ipynb

'''
from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem

# the state we desire to find is '11'
good_state = ['11']

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# define Grover's algorithm
problem = AmplificationProblem(oracle, is_good_state=good_state)

problem.grover_operator.decompose().draw(output='mpl')

from qiskit.algorithms import Grover
from qiskit.primitives import Sampler

grover = Grover(sampler=Sampler())
result = grover.amplify(problem)
print('Result type:', type(result))
print()
print('Success!' if result.oracle_evaluation else 'Failure!')
print('Top measurement:', result.top_measurement)
print('nnn iii', len(result.iterations), type(result.iterations),result.iterations)

print('OK', type(result))
qc = grover.construct_circuit(problem, power=3, measurement=True)
print('ideal circ')
print(qc)

# transpile it for the ideal  backend
from qiskit import  transpile, Aer
backend=Aer.get_backend("qasm_simulator")
qcT = transpile(qc, backend=backend)
print('transpiled  circ for ',backend)
print(qcT)
