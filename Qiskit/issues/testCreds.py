#!/usr/bin/env python3
import os
from qiskit_ibm_runtime import QiskitRuntimeService
token= os.getenv('QISKIT_IBM_TOKEN')
channel="ibm_quantum_platform"
instance='research-us'; backName='ibm_fez'  # US
#instance='research-eu'; backName='ibm_aachen'  # EU
print('Set IBM creds use  instance:%s  channel:%s  token:%s...'%(instance,channel,token[:5]))
service = QiskitRuntimeService(token=token,channel=channel,instance=instance)
# , set_as_default=True)
backends = service.backends()
print(backends)

print('\n access  %s backend ...'%backName)
backend2 = service.backend(backName)
print('use backend =', backend2.name )


