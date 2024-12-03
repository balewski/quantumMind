#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
    start_date = datetime(2024, 8, 8)  # Start date (YYYY, MM, DD)
    end_date = datetime(2024, 12, 27)   # End date (YYYY, MM, DD)

Backend             Quantum Time (min)       Total Shots (M)     Total Circuits      
-------------------------------------------------------------------------------------
ibm_kyiv            0.6                      0.1                 21                  
ibm_torino          34.9                     6.2                 1125                
ibm_brisbane        2.6                      0.5                 66                  
ibm_sherbrooke      0.5                      0.1                 21                  
ibm_kyoto           0.5                      0.1                 32
'''

from pprint import pprint
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

def fetch_jobs(service, start_date, end_date):
    """
    Fetch all jobs from the Qiskit Runtime Service within a specific date range.
    """
    jobs = service.jobs(created_after=start_date, created_before=end_date, limit=None)
    return jobs

def filter_and_process_jobs(jobs):
    """
    Filter jobs with status 'DONE' and calculate total quantum seconds, shots, and circuits per backend.
    """
    backend_data = {}
    accepted_jobs = []
    processed_jobs_count = 0

    for job in jobs:
        # Check job status
        job_status = str(job.status())
        if 'DONE' not in job_status:
            continue

        backend_name = job.backend().name
        job_metrics = job.metrics()

        # Quantum seconds
        quantum_seconds = job_metrics['usage']['quantum_seconds']
        timestamp_running = job_metrics['timestamps']['running']

        # Job results and calculations
        job_result = job.result()
        total_shots = 0
        #print('work on jid',job.job_id(),timestamp_running)
        total_circuits = len(job_result)  # Number of circuits in the job
        for result in job_result:
            # Extract the number of shots
            for key, value in vars(result.data).items():
                total_shots += value.num_shots
                break  # There is always one item, so break after the first

        # Update backend data
        if backend_name not in backend_data:
            backend_data[backend_name] = {'total_quantum_seconds': 0, 'total_shots': 0, 'total_circuits': 0}
        
        backend_data[backend_name]['total_quantum_seconds'] += quantum_seconds
        backend_data[backend_name]['total_shots'] += total_shots
        backend_data[backend_name]['total_circuits'] += total_circuits

        # Add job details to the accepted jobs list
        accepted_jobs.append({
            'job_id': job.job_id(),
            'backend': backend_name,
            'quantum_seconds': quantum_seconds,
            'timestamp_running': timestamp_running,
            'total_shots': total_shots,
            'total_circuits': total_circuits
        })

        # Print progress every 5 jobs processed
        processed_jobs_count += 1
        if processed_jobs_count % 5 == 0:
            print("Processed %d jobs with status 'DONE' so far..." % processed_jobs_count)

    return backend_data, accepted_jobs

def print_backend_summary(backend_data):
    """
    Print the backend summary in a neat table format using C-style formatting.
    """
    print("\n=== Backend Data Summary ===")
    print("%-20s%-25s%-20s%-20s" % ("Backend", "Quantum Time (min)", "Total Shots (M)", "Total Circuits"))
    print("-" * 85)
    
    for backend, data in backend_data.items():
        # Convert quantum seconds to minutes and shots to millions
        quantum_time_minutes = data['total_quantum_seconds'] / 60
        total_shots_million = data['total_shots'] / 1_000_000
        
        # Print data in C-style formatting
        print("%-20s%-25.1f%-20.1f%-20d" % (backend, quantum_time_minutes, total_shots_million, data['total_circuits']))

def main():
    # Define the date range
    start_date = datetime(2024, 8, 8)  # Start date (YYYY, MM, DD)
    end_date = datetime(2024, 12, 27)   # End date (YYYY, MM, DD)

    # Load the Qiskit Runtime Service
    service = QiskitRuntimeService(channel="ibm_quantum")

    # Fetch all jobs
    print("Fetching jobs from %s to %s..." % (start_date, end_date))
    jobs = fetch_jobs(service, start_date, end_date)
    print('M: found%d jobs of all kinds'%len(jobs))
    
    # Filter and process jobs
    backend_data, accepted_jobs = filter_and_process_jobs(jobs)

    if 0:  # very large output
        print("\n=== Accepted Jobs ===")
        pprint(accepted_jobs)

    # Output results
    print_backend_summary(backend_data)
    

if __name__ == "__main__":
    main()
