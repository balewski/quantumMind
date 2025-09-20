#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Track and analyze IBM Quantum usage costs and quantum time consumption by backend

    start_date = datetime(2024, 8, 8)  # Start date (YYYY, MM, DD)
    end_date = datetime(2024, 12, 27)   # End date (YYYY, MM, DD)

=== Backend Data Summary ===
Backend             Quantum Time (min)       Total Shots (M)     Total Circuits      Job Count           
---------------------------------------------------------------------------------------------------------
ibm_kyiv            0.6                      0.1                 21                  1                   
ibm_torino          12.5                     2.0                 528                 26                  
ibm_brisbane        2.6                      0.5                 66                  4                   
ibm_sherbrooke      0.5                      0.1                 21                  1                   

=== Monthly Aggregated Data ===
Month     Quantum Time (min)       Job Count           
------------------------------------------------------------
2024-11   16.2                     32

'''

import argparse, os
from collections import defaultdict
from pprint import pprint
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime, timedelta


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
    monthly_data = defaultdict(lambda: {"quantum_seconds": 0, "job_count": 0})
    backend_job_count = defaultdict(int)
    accepted_jobs = []
    processed_jobs_count = 0

    for i,job in enumerate(jobs):
        # Check job status
        job_status = str(job.status())
        if 'DONE' not in job_status:  continue

        backend_name = job.backend().name
        backend_job_count[backend_name] += 1

        job_metrics = job.metrics()
        quantum_seconds = job_metrics['usage']['quantum_seconds']
        timestamp_running = job_metrics['timestamps']['running']
        job_date = datetime.strptime(timestamp_running.split("T")[0], "%Y-%m-%d")
        job_month = job_date.strftime("%Y-%m")

        # Job results and calculations
        job_result = job.result()
        total_shots = 0
        total_circuits = len(job_result)  # Number of circuits in the job
        for result in job_result:
            for key, value in vars(result.data).items():
                total_shots += value.num_shots
                break  # There is always one item, so break after the first

        # Update backend data
        if backend_name not in backend_data:
            backend_data[backend_name] = {'total_quantum_seconds': 0, 'total_shots': 0, 'total_circuits': 0}

        backend_data[backend_name]['total_quantum_seconds'] += quantum_seconds
        backend_data[backend_name]['total_shots'] += total_shots
        backend_data[backend_name]['total_circuits'] += total_circuits

        # Update monthly data
        monthly_data[job_month]["quantum_seconds"] += quantum_seconds
        monthly_data[job_month]["job_count"] += 1

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
            print("Processed %d jobs with status 'DONE' so far, any jobs %d..." % (processed_jobs_count,i))

    return backend_data, monthly_data, backend_job_count, accepted_jobs


def print_backend_summary(backend_data, backend_job_count):
    """
    Print the backend summary in a neat table format using C-style formatting.
    """
    print("\n=== Backend Data Summary ===")
    print("%-20s%-25s%-20s%-20s%-20s" % (
        "Backend", "Quantum Time (min)", "Total Shots (M)", "Total Circuits", "Job Count"))
    print("-" * 105)

    for backend, data in backend_data.items():
        quantum_time_minutes = data['total_quantum_seconds'] / 60
        total_shots_million = data['total_shots'] / 1_000_000
        print("%-20s%-25.1f%-20.1f%-20d%-20d" % (
            backend, quantum_time_minutes, total_shots_million, data['total_circuits'], backend_job_count[backend]))


def print_monthly_summary(monthly_data):
    """
    Print the monthly summary in a neat table format using C-style formatting.
    """
    print("\n=== Monthly Aggregated Data ===")
    print("%-10s%-25s%-20s" % ("Month", "Quantum Time (min)", "Job Count"))
    print("-" * 60)

    for month, data in sorted(monthly_data.items()):
        quantum_time_minutes = data['quantum_seconds'] / 60
        print("%-10s%-25.1f%-20d" % (month, quantum_time_minutes, data['job_count']))

def validate_input(start_date, end_date):
    """
    Validate that the start_date is not earlier than 2024-10-30 and that start_date <= end_date.
    """
    min_start_date = datetime(2024, 10, 15)
    if start_date < min_start_date:
        raise ValueError("Start date cannot be earlier than 2024-10-30.")
    if start_date > end_date:
        raise ValueError("Start date cannot be later than end date.")

#=================================
#  M A I N
#=================================

if __name__ == "__main__":
        
    today = datetime.today()  # Get today's date    
    # Calculate the default start and end dates
    default_end_date = default_end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    default_start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process IBM Quantum jobs data.")
    parser.add_argument("--start_date", type=str, default=default_start_date, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, default=default_end_date, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))

    # Convert string arguments to datetime
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    validate_input(start_date, end_date)

    # Load the Qiskit Runtime Service
    ibmInstance=os.getenv('QISKIT_IBM_INSTANCE') # is set externaly and used by 'service'
    service = QiskitRuntimeService(channel="ibm_quantum")

    # Fetch all jobs
    print("Fetching jobs from %s to %s instance=%s  ..." % (start_date, end_date,ibmInstance))
    jobs = fetch_jobs(service, start_date, end_date)
    print('found %d any jobs'%(len(jobs))) 
    # Filter and process jobs
    backend_data, monthly_data, backend_job_count, accepted_jobs = filter_and_process_jobs(jobs)

    # Output results
    print("%d 'DONE' jobs from %s to %s..." % (len(accepted_jobs),start_date, end_date))
    
    print_backend_summary(backend_data, backend_job_count)
    print_monthly_summary(monthly_data)
