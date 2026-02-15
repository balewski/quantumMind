#!/usr/bin/env python3
"""
Retrieve simulation results from Quantinuum Nexus by job ID.

Usage:
    python3 retrieve_circuit.py <execute-job-id>
    python3 retrieve_circuit.py <execute-job-id> --project my-project
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
from collections import Counter
from pprint import pprint

import qnexus as qnx


def retrieve_results(job_id, project_name):
    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)

    ref_exec = qnx.jobs.get(id=job_id)
    status = qnx.jobs.status(ref_exec)
    print(f"\n=== Job {job_id} ===")
    print(f"  status: {status}")

    results = qnx.jobs.results(ref_exec)
    print(f"  results: {len(results)} program(s)\n")

    for idx, res_item in enumerate(results):
        print(f"--- Program {idx} ---")
        res_data = res_item.download_result()

        # pytket BackendResult with get_counts()
        if hasattr(res_data, "get_counts"):
            counts = res_data.get_counts()
            print("Result counts:")
            pprint(counts)

        # Guppy QsysResult with per-shot entries
        elif hasattr(res_data, "results"):
            print(f"{len(res_data.results)} shots:")
            for i, shot in enumerate(res_data.results):
                entries = {name: val for name, val in shot.entries}
                print(f"  shot {i}: {entries}")

            # Aggregate counts
            counts = Counter()
            for shot in res_data.results:
                key = tuple(f"{name}={val}" for name, val in shot.entries)
                counts[key] += 1
            print("\nCounts:")
            for key, cnt in counts.most_common():
                print(f'  {", ".join(key)}: {cnt}')

        else:
            print(f"  Unknown result type: {type(res_data)}")
            pprint(res_data)

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve simulation results from Nexus execute job.")
    parser.add_argument("job_id",
                        help="UUID of the Nexus execute job")
    parser.add_argument("--project", default="feb-guppy",
                        help="Nexus project name (default: feb-guppy)")
    args = parser.parse_args()
    retrieve_results(args.job_id, args.project)
