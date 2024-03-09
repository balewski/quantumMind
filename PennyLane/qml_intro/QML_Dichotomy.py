#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
This code demonstrates a quantum machine learning (QML) framework for binary classification using a parametrized quantum circuit.

'''
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from time import time
from Util_PennyLane_train import LearningRateScheduler, TrainingMonitor

#............................
#............................
#............................
class Trainer_Dichotomy():
#...!...!....................
    def __init__(self, md,XYTVT):
        X_train, Y_train, X_val, Y_val,  X_test, Y_test=XYTVT
        self.meta=md
        self.bigD={}
        cmd=md['circuit']
        self.n_qubits=cmd['num_qubit']
        self.device = qml.device('default.qubit', wires=self.n_qubits)
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test
        self.monitor = TrainingMonitor()
        
        ocf=md['opt_conf']
        #self.lr_scheduler = LearningRateScheduler(ocf['initial_step_size'],**ocf['lr_schedule'])
        
        self.lr_scheduler = LearningRateScheduler(ocf['initial_step_size'],ocf['initial_momentum'],**ocf['lr_schedule'])

#...!...!....................
    def summary(self):
        self.monitor.summary(self)
        bigD=self.bigD
        bigD['X_train']=self.X_train
        bigD['Y_train']=self.Y_train
        bigD['X_val']=self.X_val
        bigD['Y_val']=self.Y_val
        bigD['X_test']=self.X_test
        bigD['Y_test']=self.Y_test      
        
#...!...!....................
    def state_prep_circ(self, x): 
        # uses amplitude encoding 
        a=np.arccos(x)
        qml.RY(a[0], wires=0)
        qml.RY(a[1], wires=1)

#...!...!....................
    def CPhase_ansatz(self, params):
        """
        An example ansatz function using parameterized CZ gates.
        This method assumes `params` has the shape (layers, n_qubits + n_qubits - 1).
        """
        layers = params.shape[0]
        n_qubits = self.n_qubits

        for layer in range(layers):
            qml.Barrier()
            # Apply single-qubit rotations
            for qubit in range(n_qubits):
                qml.RX(params[layer, qubit], wires=qubit)
                #qml.RZ(params[layer, qubit], wires=qubit)
            
            # Apply parameterized CZ gates between adjacent qubits
            for qubit in range(n_qubits - 1):
                qml.CZ(wires=[qubit, qubit + 1])
                qml.RZ(params[layer, n_qubits + qubit], wires=qubit + 1)
                qml.CZ(wires=[qubit, qubit + 1])
                
#...!...!....................
    def efficient_SU2_ansatz(self,params):
        """
        Parameters should have a shape of (layers, n_qubits, nUang),
        where layers is the number of ansatz layers,
        n_qubits is the number of qubits 
        nUang represents the rotation angles for RX, RY, and RZ gates.
        """
        # Number of layers
        layers = params.shape[0]
        n_qubits = self.n_qubits
        for layer in range(layers): # Apply rotational gates to each qubit
            qml.Barrier()
            for qubit in range(n_qubits):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)

            # Apply CNOTs for entanglement, forming a ring of qubits
            for qubit in range(n_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
                if n_qubits<3: break  # ring requires at least 3 nodes

#...!...!....................
    def circuit(self, params):
        ansatzN=self.meta['opt_conf']['ansatz_name']
        assert ansatzN in ['CPhase','EffiSU2']
        self.meta['circuit']['ansatz_name']=ansatzN
        @qml.qnode(self.device)
        def qnode(x):
            self.state_prep_circ(x)
            if ansatzN=='CPhase':
                self.CPhase_ansatz(params)
            else:
                self.efficient_SU2_ansatz(params)
            return qml.expval(qml.PauliZ(0))
        self.monitor.log_circ_execution()
        return qnode

    
#...!...!..................
    def cost_function(self, params, X, Y):  # used by back-prop,  vectorized
        predictions = np.array([self.circuit(params)(x) for x in X])        
        cost = np.mean((Y - predictions) ** 2) # square_loss(labels, predictions)
        return cost

    
#...!...!..................
    def accuracy_metric(self, params,X, Y): # only monitoring, not used by back-prop
        pred = self.circuit(params)(  X.T)
        #...... for binary classification        
        pred_classes = [1 if p > 0 else -1 for p in pred]
        correct = np.mean(np.array(pred_classes) == Y)
        return correct

    
#...!...!....................
    def train(self):
        #.... setup
        cmd=self.meta['circuit']
        tmd=self.meta['train']
        ocf=self.meta['opt_conf']
        lrcf=ocf['lr_schedule']

        opt = NesterovMomentumOptimizer(stepsize=ocf['initial_step_size'], momentum=ocf['initial_momentum'])
        params=np.random.random(size=cmd['param_shape'])

        cc=self.circuit(params)
        print(qml.draw(cc, decimals=2)(self.X_train[0]), '\n')
        
        nSample=self.X_train.shape[0]
        T0=time()
        for it in range(tmd['num_step']):
            idxL = np.random.choice(range(nSample), size=ocf['batch_size'], replace=False)
            X_batch, Y_batch = self.X_train[idxL], self.Y_train[idxL]
            # ..... this line does the brack-propagation:
            params = opt.step(lambda p: self.cost_function(p, X_batch, Y_batch), params)

            if it % lrcf['steps_skip'] == 0:
                val_acc = self.accuracy_metric(params, self.X_val, self.Y_val)
                KK=8 # sample only 1/KK of train data to speed it up
                idxL = np.random.choice(range(nSample), size=nSample//KK, replace=False)                
                X_batch, Y_batch = self.X_train[idxL], self.Y_train[idxL]
                train_acc = self.accuracy_metric(params,X_batch , Y_batch)
                
                self.monitor.log_accuracy(it, train_acc, val_acc,self.lr_scheduler.step_size,self.lr_scheduler.momentum)
                self.monitor.update_best_params(val_acc, params,it)

                if self.lr_scheduler.should_reduce_lr(val_acc, it):
                    new_lr, new_momentum = self.lr_scheduler.adjust_learning_rate_and_momentum()
                    opt = NesterovMomentumOptimizer(stepsize=new_lr, momentum=new_momentum)
                    
            if self.lr_scheduler.check_early_stopping():                
                break  # Exit the training loop
        T1=time()
        tmd['duration']=T1-T0
        
#...!...!..................
    def infere(self, params, X):  # vectorized code
        predictions = np.array([self.circuit(params)(x) for x in X])        
        return predictions
                
#...!...!....................
    def predict_test(self):
        params=self.bigD['best_weights']
        test_acc = self.accuracy_metric(params, self.X_test, self.Y_test)
        print('\npredict: test_accuracy:%.3f\n'%(test_acc ))
        bmd=self.meta['train']['best']
        bmd['test_acc']=float(test_acc)
        
#............................
#........E.N.D...............
#............................

