#!/usr/bin/env python3
''' problem: ??
'''
from pprint import pprint
from time import time
from qiskit import QuantumCircuit

from braket.aws import AwsDevice
import qbraid  # circuit converter

#...!...!....................
def ghzCirc(nq=2):
    """Create a GHZ state preparation circuit."""
    qc = QuantumCircuit(nq, nq)
    qc.h(0)
    for i in range(1, nq):
        qc.cx(0, i)
    qc.measure(range(nq), range(nq))
    return qc

#=================================
#        M A I N
#=================================
if __name__ == "__main__":

    #device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
    device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

    print('M: device',device)

    qc = ghzCirc(4)
    print("\nQiskit Circuit:")
    print(qc)

    qc_bk=qbraid.circuit_wrapper(qc).transpile("braket")


    print("\nAmazon Braket Circuit:")
    print(qc_bk)


    shots=10

    job = device.run(qc_bk, shots=shots)
    print('\nJOB:',job)

    jobMD=job.metadata()
    
    print('\nMETA:'); pprint(jobMD)

    jobRes=job.result()

    print('M: comp circ',jobRes.get_compiled_circuit())

    print('M: counts',jobRes.measurement_counts)
  
