#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from pprint import pprint
from qiskit_ibm_provider import IBMProvider

import os

hub='ibm-q-ornl'; group='lbnl'; project='chm170'; backName = "ibmq_jakarta"

hgp = f"{hub}/{group}/{project}"

print('\nM:Try provider hgp:', hgp, 'backend:',backName)

print('M:my ~ ',os.path.join(os.path.expanduser("~")))

# token is saved inside your environment
# check you can use this command
print('IBMProvider.saved_accounts():',IBMProvider.saved_accounts())

if 0:  # special case
    myToken='bdce79a....2e9a2bdf'
    # Save token if necessary
    IBMProvider.save_account(token=myToken,overwrite=True)
    print('token saved')

# Get our backend
provider = IBMProvider()
print('M:got provider, my hgp:', hgp)

print('\n  provider backends:'); pprint(provider.backends())
backend = provider.get_backend(backName, instance=hgp)

print('\n\n access backend=%s and inspect its properties'%backName)

print('\nmy backend=',backend)
print(backend.status().to_dict())
print('propertioes keys:',backend.properties().to_dict().keys())
hw_qubits=backend.properties().to_dict() #'qubits')#
print('num hw qubits=',len(hw_qubits),hw_qubits.keys())
hw_config=backend.configuration().to_dict()
print('configuration keys:',hw_config.keys(),'n_qubits=',hw_config['n_qubits'],backend.configuration().n_qubits)
print('M:OK')
