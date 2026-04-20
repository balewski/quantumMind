#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import json
from time import time

import qnexus as qnx
from qnexus.models.references import ExecuteJobRef

# emu_330047 , 1k shots, takes 7 sec
job_ref_json = '''{"id":"0f918c67-ee36-42a8-94d4-94a767bccd0c","annotations":{"name":"exec_380c984bcf3267eb","description":"","properties":{},"created":"2026-04-08 20:44:36.665722+00:00","modified":"2026-04-08 20:44:36.665722+00:00"},"job_type":"execute","last_status":"SUBMITTED","last_message":"","last_status_detail":null,"project":{"id":"743a0fe0-7112-41a9-931d-8120926b927a","annotations":{"name":"mar_guppy2","description":null,"properties":{},"created":"2026-03-28 18:42:06.181990+00:00","modified":"2026-03-28 18:42:06.181990+00:00"},"contents_modified":"2026-04-08 18:02:11.286154+00:00","archived":false,"type":"ProjectRef"},"system":null,"backend_config_store":{"type":"QuantinuumConfig","device_name":"Helios-1E","simulator":"state-vector","machine_debug":false,"attempt_batching":false,"allow_implicit_swaps":true,"postprocess":false,"noisy_simulation":true,"target_2qb_gate":null,"user_group":"CHM170","max_batch_cost":2000,"compiler_options":{"max-qubits":11},"no_opt":true,"allow_2q_gate_rebase":false,"leakage_detection":false,"simplify_initial":false,"max_cost":2000,"error_params":null},"type":"ExecuteJobRef"}'''

# emu_ebad51 , 10k shots, takes 70 sec
job_ref_json = '''{"id":"e8d8a665-0fe4-4711-ba77-7c674d4b209d","annotations":{"name":"exec_ad7cc446b25a5b48","description":"","properties":{},"created":"2026-04-08 20:58:05.784211+00:00","modified":"2026-04-08 20:58:05.784211+00:00"},"job_type":"execute","last_status":"SUBMITTED","last_message":"","last_status_detail":null,"project":{"id":"743a0fe0-7112-41a9-931d-8120926b927a","annotations":{"name":"mar_guppy2","description":null,"properties":{},"created":"2026-03-28 18:42:06.181990+00:00","modified":"2026-03-28 18:42:06.181990+00:00"},"contents_modified":"2026-04-08 20:58:02.905068+00:00","archived":false,"type":"ProjectRef"},"system":null,"backend_config_store":{"type":"QuantinuumConfig","device_name":"Helios-1E","simulator":"state-vector","machine_debug":false,"attempt_batching":false,"allow_implicit_swaps":true,"postprocess":false,"noisy_simulation":true,"target_2qb_gate":null,"user_group":"CHM170","max_batch_cost":2000,"compiler_options":{"max-qubits":11},"no_opt":true,"allow_2q_gate_rebase":false,"leakage_detection":false,"simplify_initial":false,"max_cost":2000,"error_params":null},"type":"ExecuteJobRef"}'''

# emu_fe7a48 , 10k shots, takes 70 sec
job_ref_json = '''{"id":"06a50482-3566-4e8c-80b2-06fae0996270","annotations":{"name":"exec_e69518ac4b46bb31","description":"","properties":{},"created":"2026-04-08 20:57:26.035547+00:00","modified":"2026-04-08 20:57:26.035547+00:00"},"job_type":"execute","last_status":"SUBMITTED","last_message":"","last_status_detail":null,"project":{"id":"743a0fe0-7112-41a9-931d-8120926b927a","annotations":{"name":"mar_guppy2","description":null,"properties":{},"created":"2026-03-28 18:42:06.181990+00:00","modified":"2026-03-28 18:42:06.181990+00:00"},"contents_modified":"2026-04-08 20:45:10.383906+00:00","archived":false,"type":"ProjectRef"},"system":null,"backend_config_store":{"type":"QuantinuumConfig","device_name":"Helios-1E","simulator":"state-vector","machine_debug":false,"attempt_batching":false,"allow_implicit_swaps":true,"postprocess":false,"noisy_simulation":true,"target_2qb_gate":null,"user_group":"CHM170","max_batch_cost":2000,"compiler_options":{"max-qubits":11},"no_opt":true,"allow_2q_gate_rebase":false,"leakage_detection":false,"simplify_initial":false,"max_cost":2000,"error_params":null},"type":"ExecuteJobRef"}'''

def main():
    data = json.loads(job_ref_json)
    ref_exec = ExecuteJobRef(**data)

    print('\nstatus3:', qnx.jobs.status(ref_exec))

    t0 = time()
    results = qnx.jobs.results(ref_exec)
    print('results() elaT=%.2f sec' % (time() - t0))

    nCirc = len(results)
    print('nCirc=%d' % nCirc)

    countsL = [None] * nCirc
    t0 = time()
    for ic in range(nCirc):
        result_nx = results[ic].download_result()
        print('downloaded result for circ %d, elaT=%.2f processing...' % (ic, time() - t0))
        counts_nx = result_nx.collated_counts()
        bitstring_dict = {"".join(tag_res[1] for tag_res in key)[::-1]: val for key, val in counts_nx.items()}
        countsL[ic] = bitstring_dict

    print('retrieved %d circuits' % len(countsL))
    if nCirc > 0:
        print('first counts:', countsL[0])


if __name__ == "__main__":
    main()
