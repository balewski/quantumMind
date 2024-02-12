# Daniel Mills from TKet team: I'm including here some examples of some of the things that I've said above.  
from qermit.spam import gen_FullyCorrelated_SPAM_MitRes, gen_UnCorrelated_SPAM_MitRes
from pytket.circuit import Node
from pytket import Circuit
from qermit import CircuitShots
from qermit.taskgraph import gen_compiled_MitRes
from pytket.extensions.quantinuum import QuantinuumBackend


# As you observe, there is an issue with SPAM that is repaired 
# by a fix to pytket-quantinuum in the next release. As this
# is the case, for illustrative purposes I will only use the
# syntax checker for now.
backend = QuantinuumBackend(device_name='H1-1SC')

shots=2000
uc_spam_mitres = gen_UnCorrelated_SPAM_MitRes(
   backend = backend,
   calibration_shots = shots*2
)

compile_mitres = gen_compiled_MitRes(backend = backend)

test_c_0 = Circuit(4).X(0).X(2).measure_all()
test_c_1 = Circuit(4).X(1).X(3).measure_all()

test_experiment = [
    CircuitShots(Circuit = test_c_0, Shots = shots),
    CircuitShots(Circuit = test_c_1, Shots = shots)
]

# An initial run of the circuits above for comparison purposes.
# This should add 2 jobs to the queue
basic_results = compile_mitres.run(test_experiment)

print(basic_results[0].get_counts())
print(basic_results[1].get_counts())

# A SPAM mitigated run. This will add 2 circuits to the queue
# for calibration purposes (the all 1 and all 0 state preparation
# and measurement) and will add the two circuits above to the
# queue.
spam_mitigated_results = uc_spam_mitres.run(test_experiment)

print(spam_mitigated_results[0].get_counts())
print(spam_mitigated_results[1].get_counts())

# Note that running this a second time will add the two
# circuits of interest to the queue, but will not
# rerun the calibration
spam_mitigated_results = uc_spam_mitres.run(test_experiment)

print(spam_mitigated_results[0].get_counts())
print(spam_mitigated_results[1].get_counts())

# Alternatively one can use gen_FullyCorrelated_SPAM_MitRes
# and specify the qubits to characterise. This will
# again add two circuits, but they will be much smaller.
shots=2000
fc_spam_mitres = gen_FullyCorrelated_SPAM_MitRes(
    backend = backend,
    calibration_shots = shots*2,
    correlations = [[Node(index=i)] for i in range(4)]
)

spam_mitigated_results = fc_spam_mitres.run(test_experiment)

print(spam_mitigated_results[0].get_counts())
print(spam_mitigated_results[1].get_counts())
