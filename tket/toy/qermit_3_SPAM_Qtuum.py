#!/usr/bin/env python3
# https://cqcl.github.io/Qermit/manual/manual_mitres.html

#...!...!....................
def activate_qtuum_api():
    from pytket.extensions.quantinuum.backends.credential_storage import  MemoryCredentialStorage
    from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
    import os

    MY_QTUUM_NAME=os.environ.get('MY_QTUUM_NAME')
    MY_QTUUM_PASS=os.environ.get('MY_QTUUM_PASS')
    print('credentials MY_QTUUM_NAME=',MY_QTUUM_NAME)
    cred_storage = MemoryCredentialStorage()
    cred_storage._save_login_credential(user_name=MY_QTUUM_NAME, password=MY_QTUUM_PASS)
    api = QuantinuumAPI(token_store = cred_storage)
    return api

print(' ========== MAIN ======== ')

print('M:start...')
from pytket.extensions.quantinuum import QuantinuumBackend

machine = 'H1-1E'
api=activate_qtuum_api()
backend = QuantinuumBackend(device_name=machine, api_handler=api)
print('machine=',machine)
print("status:", backend.device_state(device_name=machine, api_handler=api))

shots=2000
# ----  configure MitRes to use Qtuum backend
from qermit.spam import gen_UnCorrelated_SPAM_MitRes
from pytket.extensions.qiskit import IBMQEmulatorBackend

uc_spam_mitres = gen_UnCorrelated_SPAM_MitRes(
   backend = backend,
   calibration_shots = shots*2
)
uc_spam_mitres.get_task_graph()


from qermit.taskgraph import gen_compiled_MitRes

compile_mitres = gen_compiled_MitRes(backend = backend)
compile_mitres.get_task_graph()


from pytket import Circuit
from qermit import CircuitShots

test_c_0 = Circuit(4).X(0).X(2).measure_all()
test_c_1 = Circuit(4).X(1).X(3).measure_all()

test_experiment = [CircuitShots(Circuit = test_c_0, Shots = shots), CircuitShots(Circuit = test_c_1, Shots = shots)]
basic_results = compile_mitres.run(test_experiment)

print('raw H1-1E results:')
print(basic_results[0].get_counts())
print(basic_results[1].get_counts())

print('====== apply SPAM mitigation')
spam_mitigated_results = uc_spam_mitres.run(test_experiment)
print(spam_mitigated_results[0].get_counts())
print(spam_mitigated_results[1].get_counts())
