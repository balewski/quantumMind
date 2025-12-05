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
#from qiskit_ionq import IonQProvider
import os
import requests
from time import sleep
import argparse


def get_ionq_metadata(jid: str) -> dict:
    """Retrieve job metadata from the IonQ API."""
    # Constants
    API_BASE_URL = "https://api.ionq.co/v0.3/jobs"
    IONQ_API_KEY = os.getenv("IONQ_API_KEY")
    HEADERS = {
        "Authorization": f"apiKey {IONQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.get(f"{API_BASE_URL}/{jid}", headers=HEADERS)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Retrieve IonQ job metadata.")
    parser.add_argument("--jobId", 
                        default="019a8e63-5bc4-7048-8e79-62439297d1c4", 
                        help="The IonQ job ID to retrieve metadata for.")
    args = parser.parse_args()

    print(f"Retrieving metadata for JID: {args.jobId}")
    try:
        jmeta = get_ionq_metadata(args.jobId)       
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    #pprint(jmeta)
    
    outD = {
        'qubits': jmeta.get('qubits'),
        'shots': jmeta.get('shots'),
        'start_time': jmeta.get('start'),
        'status': jmeta.get('status'),
        'target_qpu': jmeta.get('target'),
        'type': jmeta.get('type'),
        'num_circ': jmeta.get('circuits'),
        'cost_billable_time_us': jmeta.get('cost_billable_time_us'),
        'error_mitigation': jmeta.get('error_mitigation'),
        'exec_time': jmeta.get('execution_time') / 1_000 if jmeta.get('execution_time') is not None else None, # Convert us to seconds
        'gate_counts': jmeta.get('gate_counts'),
    }
    pprint(outD)
    print('M:and')
    
if __name__ == "__main__":
    main()

