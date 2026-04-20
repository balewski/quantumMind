#!/usr/bin/env python3

# taken from https://docs.aws.amazon.com/braket/latest/developerguide/braket-pricing.html

#import any required modules
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.tracking import Tracker
from pprint import pprint

#create our bell circuit
circ = Circuit().h(0).cnot(0,1)
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
with Tracker() as tracker:
    task = device.run(circ, shots=10).result()

#Your results
print(task.measurement_counts)
print('cost tracker, charge/$:',tracker.simulator_tasks_cost())
pprint(tracker.quantum_tasks_statistics())
