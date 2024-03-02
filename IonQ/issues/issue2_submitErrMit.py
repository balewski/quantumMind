#!/usr/bin/env python3
''' 
fire Aria-1 job w/o error mitigation
https://cloud.ionq.com/jobs

How to use debiasing
https://ionq.com/resources/debiasing-and-sharpening

The code:  https://github.com/qiskit-community/qiskit-ionq/blob/main/qiskit_ionq/constants.py

'''
import qiskit as qk
from pprint import pprint
from qiskit_ionq import IonQProvider, ErrorMitigation
import os
import requests
from time import sleep

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

#...!...!....................
def make_ghz_circ(nq):
    name='ghz_%dq'%nq
    ghz = qk.QuantumCircuit(nq, nq,name=name)
    ghz.h(0)
    ghz.h(nq-1)
    for idx in range(1,nq):
        ghz.cx(0,idx)
    
    ghz.barrier(range(nq))
    ghz.measure(range(nq), range(nq))
    print(ghz)
    return ghz

#=================================
#  M A I N 
#=================================
if __name__ == "__main__":

    provider = IonQProvider()   # Remember to set env IONQ_API_KEY='....'   
    backName='aria-1'
    backend = provider.get_backend("ionq_qpu."+backName)
    print('\nmy backend=',backend)

    shots=7000
    
    # -------Create a Quantum Circuit
    qc=make_ghz_circ(5)
    job= backend.run(qc,shots=shots, error_mitigation =ErrorMitigation.NO_DEBIASING)
    
    jid=job.job_id()
    print('submitted JID',jid)
    sleep(5) # wait for instantiation of meta-data
   
    print("rawMD:")
    jmeta = get_metadata(jid)
    pprint(jmeta)
    print('\n kill JID:',jid)
    job.cancel()
    print('\n you should see "good error" : Unable to retreive result ')
        
    
    sleep(5) # wait for job to die
    jstat=job.status()
    print('M: status=',jstat)
