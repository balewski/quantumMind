#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
This code demonstrates a quantum machine learning (QML) framework for binary classification using a parametrized quantum circuit.

'''
import pennylane as qml
from pennylane import numpy as np
import numpy as cnp  # Use cnp (conventional numpy) for standard numpy operations
from pennylane.optimize import NesterovMomentumOptimizer
from time import time

#............................
#............................
#............................
class LearningRateScheduler:
#...!...!..................
    def __init__(self, initial_step_size, lr_reduction_factor, acc_improvement_threshold, steps_cool,max_num_reduction, **vargs):
        self.step_size = initial_step_size
        self.lr_reduction_factor = lr_reduction_factor
        self.improvement_threshold = acc_improvement_threshold
        self.steps_cool = steps_cool  # Minimum steps between reductions
        self.max_num_reduction = max_num_reduction  # Maximum number of LR reductions before early stopping
        self.reduction_count = 0  # Track the number of times LR has been reduced
 
        self.last_reduction_step = 0
        self.best_val_accuracy = 0

#...!...!..................
    def should_reduce_lr(self, accuracy, current_step):
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
        if (accuracy < self.best_val_accuracy - self.improvement_threshold and
                current_step - self.last_reduction_step >= self.steps_cool):
            self.last_reduction_step = current_step
            return True
        return False

#...!...!..................
    def adjust_learning_rate(self):
        self.step_size *= self.lr_reduction_factor
        self.reduction_count += 1
        print("  %d reducing step size to %.4f "%(self.reduction_count,self.step_size))
        return self.step_size
    
#...!...!..................
    def check_early_stopping(self):
        earlyStop=self.reduction_count >= self.max_num_reduction
        if earlyStop:
            print("Early stopping triggered after {} LR reductions.".format(self.max_num_reduction))
        return earlyStop
    
#............................
#............................
#............................
class TrainingMonitor:
#...!...!..................
    def __init__(self):
        self.best_val_accuracy = 0
        self.best_step = 0
        self.best_fcnt = 0
        self.best_params = None
        self.circuit_executions = 0
        self.history=[]

    def update_best_params(self, accuracy, params,step):
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_params = params.copy()
            self.best_step=step
            self.best_fcnt = self.circuit_executions
            print("  best validation accuracy: %.4f  fcnt=%d" %( self.best_val_accuracy,self.circuit_executions))

#...!...!..................
    def log_circ_execution(self):
        self.circuit_executions += 1

#...!...!..................
    def log_accuracy(self, step, train_accuracy, val_accuracy,lr):
        rec=[float(step), float(train_accuracy), float(val_accuracy),float(lr)]
        print('rec:',rec)
        self.history.append(rec)
        print(f"Step: {step} | Executions: {self.circuit_executions} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f} | lr: {lr:.4f}")

#...!...!..................
    def summary(self,trainer):
        print('\nTrainingMonitor summary, best val_accuracy=%.3f '%self.best_val_accuracy)
        
        tmd=trainer.meta['train']
        bmd={};  tmd['best']=bmd
        bmd['steps']=self.best_step
        bmd['val_acc']=float(self.best_val_accuracy)
        bmd['fcnt']=self.best_fcnt

        trainer.bigD['best_weights']=self.best_params
        xx=np.array(self.history)

        trainer.bigD['train_hist']=cnp.array(xx)
         

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
        self.lr_scheduler = LearningRateScheduler(ocf['initial_step_size'],**ocf['lr_schedule'])

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
            # Apply single-qubit rotations
            for qubit in range(n_qubits):
                qml.RZ(params[layer, qubit], wires=qubit)
            
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
    def test_encoding(self):
        @qml.qnode(self.device)
        def qnode(x):
            self.state_prep_circ(x)     
            return qml.expval(qml.PauliZ(0))
 
        x=self.X_train[0]
        print('\nDICH encoder circ:',x.shape)
        print(qml.draw(qnode)(x))
    
        if 0:  # just printing
            x=self.X_train[0]
            cmd=self.meta['circuit'] 
            params=np.random.random(size=cmd['param_shape'])
            print('\nDICH full circ:',params.shape,x.shape)
            print(qml.draw(qnode)(params,x))
                    
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
    def cost_function(self, params, X, Y):  # vectorized code
        predictions = np.array([self.circuit(params)(x) for x in X])        
        cost = np.mean((Y - predictions) ** 2) # square_loss(labels, predictions)
        return cost, predictions # 2 values to save 

#...!...!..................
    def accuracy_metric(self, predictions, Y):
        # Adjusting for binary classification
        predicted_classes = [1 if p > 0 else -1 for p in predictions]
        correct = np.mean(np.array(predicted_classes) == Y)
        return correct

#...!...!..................
    def adjust_learning_rate(self, step, step_size, lr_reduction_factor, steps_since_last_reduction, M):
        # Check if at least M steps have passed since the last reduction
        if step - steps_since_last_reduction >= M:
            step_size *= lr_reduction_factor
            print(f"Step {step}: Reducing step size to {step_size:.4f}")
            steps_since_last_reduction = step
            return step_size, steps_since_last_reduction, True
        return step_size, steps_since_last_reduction, False
   
#...!...!....................
    def train(self):
        #.... setup
        cmd=self.meta['circuit']
        tmd=self.meta['train']
        ocf=self.meta['opt_conf']
        lrcf=ocf['lr_schedule']

        opt = NesterovMomentumOptimizer(ocf['initial_step_size'], momentum=0.90)
        params=np.random.random(size=cmd['param_shape'])
        #print('pp1',params.shape,type(params))
        steps_since_last_reduction = 0  # Initialize step counter for LR reduction
       
        T0=time()
        for it in range(tmd['num_step']):
            idxL = np.random.choice(range(len(self.X_train)), size=ocf['batch_size'], replace=False)
            X_batch, Y_batch = self.X_train[idxL], self.Y_train[idxL]
            #params, _ = opt.step_and_cost(lambda p: self.cost_function(p, X_batch, Y_batch)[0], params)
            params = opt.step(lambda p: self.cost_function(p, X_batch, Y_batch)[0], params)

            if it % lrcf['steps_skip'] == 0:
                _, val_predictions = self.cost_function(params, self.X_val, self.Y_val)
                val_accuracy = self.accuracy_metric(val_predictions, self.Y_val)
                KK=10# sample only 1/KK of train data to speed it up
                idxL = np.random.choice(range(len(self.X_train)), size=len(self.X_train)//KK, replace=False)                
                X_batch, Y_batch = self.X_train[idxL], self.Y_train[idxL]
                train_predictions = self.infere(params, X_batch)
                train_accuracy = self.accuracy_metric(train_predictions, Y_batch)
                
                self.monitor.log_accuracy(it, train_accuracy, val_accuracy,self.lr_scheduler.step_size)
                self.monitor.update_best_params(val_accuracy, params,it)


                if self.lr_scheduler.should_reduce_lr(val_accuracy, it):
                    self.lr_scheduler.adjust_learning_rate()
                    opt = NesterovMomentumOptimizer(self.lr_scheduler.step_size)
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
        #print('pp2',params.shape,type(params))
        test_predictions = self.infere(params, self.X_test)
        test_accuracy = self.accuracy_metric(test_predictions, self.Y_test)
        print('\npredict: test_accuracy:%.3f\n'%(test_accuracy ))
        bmd=self.meta['train']['best']
        bmd['test_acc']=test_accuracy
        
#............................
#........E.N.D...............
#............................

