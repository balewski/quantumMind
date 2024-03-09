__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


import numpy as cnp  # Use cnp (conventional numpy) for standard numpy operations

#............................
#............................
#............................
class LearningRateScheduler:
    def __init__(self, initial_step_size, initial_momentum, lr_reduction_factor, acc_improvement_threshold, steps_cool, max_num_reduction, momentum_reduction_factor, **vargs):
        self.step_size = initial_step_size
        self.lr_reduction_factor = lr_reduction_factor
        self.momentum = initial_momentum
        self.momentum_reduction_factor = momentum_reduction_factor
        self.improvement_threshold = acc_improvement_threshold
        self.steps_cool = steps_cool
        self.max_num_reduction = max_num_reduction
        self.reduction_count = 0
        
        self.last_reduction_step = 0
        self.best_val_accuracy = 0

    def should_reduce_lr(self, accuracy, current_step):
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
        if (accuracy < self.best_val_accuracy - self.improvement_threshold and
                current_step - self.last_reduction_step >= self.steps_cool):
            self.last_reduction_step = current_step
            return True
        return False

    def adjust_learning_rate_and_momentum(self):
        if self.reduction_count%2==0:
            self.step_size *= self.lr_reduction_factor
        else:
            self.momentum *= self.momentum_reduction_factor
        self.reduction_count += 1
        print("  %d reducing step size to %.4f and momentum to %.4f" % (self.reduction_count, self.step_size, self.momentum))
        return self.step_size, self.momentum
    
    def check_early_stopping(self):
        earlyStop = self.reduction_count >= self.max_num_reduction
        if earlyStop:
            print("Early stopping triggered after {} LR and momentum reductions.".format(self.max_num_reduction))
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
    def log_accuracy(self, step, train_accuracy, val_accuracy,lr,momentum):
        rec=[float(step), float(train_accuracy), float(val_accuracy),float(lr),float(momentum)]
        #print('rec:',rec)
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
        trainer.bigD['train_hist']=cnp.array(self.history)
         
