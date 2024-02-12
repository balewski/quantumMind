#!/usr/bin/env python3
# coding: utf-8

# In[345]:


# resetting the entire session
#get_ipython().run_line_magic('reset', '-f')
# Iris data-set
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target


# In[346]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape


# ### Quantum Machine Learning Model
# we’ll train a variational quantum classifier (VQC), available in Qiskit Machine Learning 
#  Two of its central elements are the feature map and ansatz.

# In[347]:


# features circuit
from qiskit.circuit.library import ZZFeatureMap
num_features = X.shape[1]
feature_circ = ZZFeatureMap(feature_dimension=num_features, reps=1)
feature_circ.decompose().draw(output='text', fold=-1)


# In[348]:


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector

def rotation_block(circuit, parameters):
    # assumes nb of params equal to num of qubits
    for i in range(circuit.num_qubits):
        circuit.ry(parameters[i],i)

def non_linear(circuit, param, qubit):
    # the circuit, the parameter and the qubit of interest
    circuit.ry(param, qubit)
    circuit.measure(qubit, 0)
    circuit.ry(-param, qubit)

def myAnsatz(nFeat, reps, midMeasLL, barrier=True, final_rotation=True):
    #... compute range of paramaters index per repetition
    # midMeasLL is list of list, different qubits can be measured in each repetition
    assert len(midMeasLL)==reps
    n1=0 
    nParL=[]
    for midmL in midMeasLL:
        n0=n1
        n1=n0+nFeat+len(midmL)
        nParL.append(n0)
    if final_rotation: 
        n0=n1
        n1=n0+nFeat
        nParL.append(n0)
    totPar=n1
    print('nParL:',nParL,totPar)
   
    beta = ParameterVector('beta', length=totPar)
    qubits = QuantumRegister(nFeat,'q')
    meas = ClassicalRegister(nFeat,'c')
    qc = QuantumCircuit(qubits, meas)

    for ir in range(reps):
        n0=nParL[ir]
        rotation_block(qc, beta[n0:n0+nFeat])
        if barrier: qc.barrier()
        
        #entanglement block, here linear but can easily be changed depending on what you want ...
        
        qc.cx(range(nFeat-2,-1,-1), range(nFeat-1,0,-1))
        if qc: qc.barrier()
    
        # non-linear part
        qL=midMeasLL[ir]
        n0+=nFeat
        print('icyc:%d qL:%s'%(ir,str(qL)))
        for j,qid in enumerate(qL):
            non_linear(qc, beta[n0+j],qid)
        if barrier: qc.barrier()
        
    #final rotation block, optional
    if final_rotation: 
        n0=nParL[-1]
        rotation_block(qc, beta[n0:n0+nFeat])
        
    if barrier: qc.barrier()
    qc.measure(range(nFeat),meas)
    return qc, qubits, meas


# Anzatz circuit
nReps=6
midMeasLL=[ [] for i in range(nReps) ] # no midMeas at all
#midMeasLL=[[2,3],[]]  # select midMeas qubits per repetition
#midMeasLL=[[],[3],[2],[0],[1],[]]  # select midMeas qubits per repetition
ansatz_circ = myAnsatz(num_features, reps=nReps, midMeasLL=midMeasLL, barrier=True, final_rotation=True)[0]
#ansatz_circ.decompose().draw(output='text', fold=-1)
print(ansatz_circ)

from matplotlib import pyplot as plt
from IPython.display import clear_output
import time

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
        
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    nIter=len(objective_func_vals)
    if nIter==1: 
        plt.tstart=time.time() 
        txt='iter=1'
    else:
        elaT = (time.time() - plt.tstart)/60.
        speed=60*(nIter)/elaT # I missed 0-th iteration
        txt='iteration:%d  elaT=%.1f (min), speed %.1f iter/h,  x-entropy=%.2f'%(nIter,elaT,speed,obj_func_eval)
    print('done:',txt)
    
    plt.title("Objective function, "+txt)
    plt.xlabel("Iteration")
    plt.ylabel("Corssentropy")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.grid(color='b', linestyle='--', linewidth=0.5)
    plt.show()


# ### Training of Qiskit NeuralNetworkClassifier 

if 1:  # backend = density matrix simulator
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
    #load the service
    service = QiskitRuntimeService(channel = 'ibm_quantum')
    backend = service.get_backend('ibmq_qasm_simulator') # change this for a real execution
    # create the program for samplig results on a backend
    options = Options()
    options.resilience_level = 0  # no need to post-process for ideal backend
    options.execution.shots =1000
    session = Session(backend=backend)
    sampler = Sampler(session=session, options=options)
    #sampler = Sampler(backend, options) #if you do not want to use sessions
else:  # backend= state vector 
    assert sum(sum(lst) for lst in midMeasLL) ==0  # it can't handle mid-circui measurements
    from qiskit.primitives import Sampler  # state vector
    sampler = Sampler()

# To make the training process faster, we choose a gradient-free optimizer.
from qiskit.algorithms.optimizers import COBYLA
nIter=170
optimizer = COBYLA(maxiter=nIter)


# construct quantum circuit  (only for inspection)
num_cregs = len(ansatz_circ.clbits) # number of classical registers in the ansatz
circuit = QuantumCircuit(num_features,num_cregs)
circuit.append(feature_circ, range(num_features))
circuit.append(ansatz_circ, range(num_features),range(num_cregs))
circuit.decompose().draw(output="text", fold=-1)


# ### SamplerQNN


# maps bitstrings to label
def recoLabel4(mval):  # mval=int(bistrings)
    x=mval%4 # --> 0,1,2,3
    y=(x+1)%4  # map 0,1,2
    return y
numLabels = 4  # corresponds to the number of classes, must cover all  po

def recoLabel(mval):  # mval=int(bistrings)
    x=mval%3 # --> 0,1,2
    return x
numLabels = 3  # corresponds to the number of classes, must cover all  possible outcomes of recoLabel(.).

# construct QNN
# see https://github.com/qiskit-community/qiskit-machine-learning/blob/main/docs/tutorials/01_neural_networks.ipynb
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN

# SamplerQNN directly consumes samples from measuring the quantum circuit, it does not require a custom observable.
sampler_qnn = SamplerQNN(
    circuit=circuit.decompose(), #we decompose the circuit because samplerqnn will check if the measurements are present in the circuit
    input_params=feature_circ.parameters,
    weight_params=ansatz_circ.parameters,
    interpret=recoLabel, # interpret  bitstrings.
    output_shape=numLabels, # must match to interpreter
    sampler=sampler
)
'''These output samples are interpreted by default as the probabilities of measuring the integer index c
orresponding to a bitstring. However, the SamplerQNN also allows us to specify an interpret function 
to post-process the samples. This function should be defined so that it takes a measured integer 
(from a bitstring) and maps it to a new value, i.e. non-negative integer.
'''

# construct classifier
model = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=optimizer, callback=callback_graph
)


# create empty array for callback to store evaluations of the objective function
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

start = time.time()
# fit classifier to data
model.fit(X_train, y_train)
elaT = (time.time() - start)/3600.
print("Training time: %.1f h, last value:%.2f"%(elaT,objective_func_vals[-1]))

# return to default figsize
plt.rcParams["figure.figsize"] = (6, 4)


# ### predict for few samples

# Predict labels for the test data
y_pred =model.predict(X_test)
# Compute the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
#print(conf_matrix)
print('\nconfusion matrix, test samples:%d'%(y_pred.shape[0]))
for i,rec  in enumerate(conf_matrix):
    print('true:%d  reco:%s'%(i,rec))


# Now we check out how well our classical model performs. 
# mean accuracy of the classifier
train_score_c4 = model.score(X_train, y_train)
test_score_c4 = model.score(X_test, y_test)

print(f"mean accuracy  on training dataset: {train_score_c4:.2f}")
print(f"mean accuracy on  test dataset:     {test_score_c4:.2f}")


# In[359]:


# run full circuit for 
weights=model.weights
'weihts:',weights.shape, weights


# In[360]:



#samples=X_test[:nSamp]
#labels=y_test[:nSamp]
#pred=model.predict(samples)
nSamp=y_test.shape[0]
nok=0
print('true,pred')
for t,p in zip(y_test, y_pred):
    print(p,t,p==t)
    nok+=p==t
print('avr prob=%.2f'%(nok/nSamp))



