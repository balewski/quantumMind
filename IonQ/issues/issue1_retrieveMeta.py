#!/usr/bin/env python3
''' problem: retrieve selected meta data from a job
Solution: scarpe the web page
'''


from qiskit_ionq import IonQProvider
from pprint import pprint
import os
import requests

# Constants
API_BASE_URL = "https://api.ionq.co/v0.3/jobs"
IONQ_API_KEY = os.getenv("IONQ_API_KEY")
HEADERS = {
    "Authorization": f"apiKey {IONQ_API_KEY}",
    "Content-Type": "application/json",
}

def get_metadata(jid: str) -> dict:
    """Retrieve job metadata from the IonQ API."""
    response = requests.get(f"{API_BASE_URL}/{jid}", headers=HEADERS)
    return response.json()


print('M: activate IonQProvider() ...')
provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'
backend = provider.get_backend("ionq_simulator")  # this is V1 backend

#jid="64374135-6dd4-462b-bb07-ff54c26abea9"  # small job, completed
jid="26b00c0d-9477-4507-8f01-d87a8b74d14d" # enabled 'debias'=error mitigation
#jid="2234c391-738c-4c57-9687-ce8a1da2d8c0" # canceled job

print('RIJ: retrieve jid:',jid)

print("rawMD:")
jmeta = get_metadata(jid)
for k in ['request', 'response', 'registers', 'predicted_execution_time', 'execution_time','gate_counts', 'error_mitigation','status','shots','circuits','qubits']:
    print(f"{k} : {jmeta.get(k, 'Not available')}")

print('M: qiskit job info')
job=backend.retrieve_job(jid)
jstat=job.status()

print('M: status=',jstat)

jobRes=job.result()
print('job  meta'); pprint(jobRes._metadata)

print('M:ok')

