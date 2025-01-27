#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import perceval as pcvl
print('perceval ver:',pcvl.__version__)
token = os.environ.get('MY_QUANDELA_TOKEN')
assert token!=None
pcvl.save_token(token)

remProc = pcvl.RemoteProcessor("qpu:ascella")  # QPU name does not matter ??

jid='0fb9b13b-8124-4b5a-a860-d587c0df6b80'  # qpu:ascella

jid='f6200941-3697-432c-ae4b-d060d542ff72'  # sim:ascella

job = remProc.resume_job(jid)
print('Job status =%s name=%s'%(job.status(),job.name))

if 0:
    job.cancel()
    job.id()
    # 'is_complete', 'is_failed', 'is_running', 'is_success', 'is_waiting', 'name', 
    

results = job.get_results()
print(results['results'])

print('M:OK',job.id)

#1print(dir(job))
''' 
'cancel', 'execute_async', 'execute_sync', 'from_id', 'get_results', 'id', 'is_complete', 'is_failed', 'is_running', 'is_success', 'is_waiting', 'name', 'rerun', 'status'
'''

print('proc name:',pcvl.remote_processor)
#print(dir(pcvl))
'''
'abstract_component', 'abstract_processor', 'algorithm', 'algorithms', 'allstate_iterator', 'anonymize_annotations', 'backends', 'build_spatial_output_states', 'canvas', 'catalog', 'circuit', 'comp_utils', 'component_catalog', 'components', 'conversion', 'convert_polarized_state', 'core_catalog', 'decompose_perms', 'density_matrix', 'density_matrix_utils', 'deprecated', 'detector', 'error_mitigation', 'feed_forward_configurator', 'format', 'format_parameters', 'generate_all_logical_states', 'generic_interferometer', 'get_basic_state_from_ports', 'get_detection_type', 'get_logger', 'get_pauli_basis_measurement_circuit', 'get_pauli_eigen_state_prep_circ', 'get_pauli_eigenvector_matrix', 'get_pauli_eigenvectors', 'get_pauli_gate', 'global_params', 'globals', 'import_module', 'job', 'job_group', 'job_status', 'linear_circuit', 'local_job', 'logging', 'logical_state', 'matrix', 'matrix_double', 'max_photon_state_iterator', 'metadata', 'mlstr', 'noise_model', 'non_unitary_components', 'parameter', 'partial_progress_callable', 'pdisplay', 'pdisplay_to_file', 'persistent_data', 'photon_recycling', 'polarization', 'port', 'post_select_distribution', 'post_select_statevector', 'postselect', 'postselect_independent', 'probs_to_sample_count', 'probs_to_samples', 'processor', 'processor_circuit_configurator', 'progress_cb', 'providers', 'qmath', 'quandela', 'random_seed', 'register_plugin', 'remote_job', 'remote_processor', 'rendering', 'rpc_handler', 'runtime', 'sample_count_to_probs', 'sample_count_to_samples', 'samples_to_probs', 'samples_to_sample_count', 'save_token', 'scaleway', 'serialization', 'session', 'simple_complex', 'simple_float', 'simulators', 'source', 'stategenerator', 'statevector', 'tensorproduct', 'tomography_exp_configurer', 'unitary_components', 'use_perceval_logger', 'use_python_logger', 'utils']
'''



