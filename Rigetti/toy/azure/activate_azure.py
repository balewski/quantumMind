#!/usr/bin/env python3
from azure.quantum.qiskit import AzureQuantumProvider

'''
Avaliable Azure workspace targets:
- ionq.qpu
- ionq.qpu.aria-1
- ionq.simulator
- quantinuum.hqs-lt-s1
- quantinuum.hqs-lt-s1-apival
- quantinuum.hqs-lt-s2
- quantinuum.hqs-lt-s2-apival
- quantinuum.hqs-lt-s1-sim
- quantinuum.hqs-lt-s2-sim
- quantinuum.qpu.h1-1
- quantinuum.sim.h1-1sc
- quantinuum.qpu.h1-2
- quantinuum.sim.h1-2sc
- quantinuum.sim.h1-1e
- quantinuum.sim.h1-2e
- rigetti.sim.qvm
- rigetti.qpu.aspen-11
- rigetti.qpu.aspen-m-2
'''

import os
  
#...!...!..................
def activate_azure_provider(backName,verb=1):
    print('First time do:  az login')
    # creds are stored on my mac at ./ssh/azure.creds
    SUBSCRIPTION_ID=os.environ.get('MY_AZURE_SUBSCRIPTION_ID')
    RESOURCE_NAME=os.environ.get('MY_AZURE_RESOURCE_NAME')
    AZURE_LOCATION="eastus"
    #print("my Azure creds:',SUBSCRIPTION_ID, RESOURCE_NAME, AZURE_LOCATION)
    from azure.quantum.qiskit import AzureQuantumProvider
    provider = AzureQuantumProvider (
        resource_id = "/subscriptions/"+SUBSCRIPTION_ID+"/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/"+RESOURCE_NAME,
        location  = AZURE_LOCATION
    )

    if verb>1:
        print("Avaliable Azure workspace targets:")
        for backend in provider.backends():
            print("- " + backend.name())

    backend = provider.get_backend(backName) 
    print('AAP: backend',backend.status())
    return backend

