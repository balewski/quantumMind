#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Exercising save/restore of various qiskit objects

1) job.results() == qiskit.result.result.Result
It is not elegant, since I need to provide an empty jobResult object and call results.from_dict(blob)


2) CompleteMeasFitter()

3) circuit  from/to qasm file


'''

# Import general libraries (needed for functions)
import numpy as np
import time,os,sys
from pprint import pprint
sys.path.append(os.path.abspath("../utils/"))
from Circ_Util import read_yaml, write_yaml, save_measFitter_Yaml, restore_measFitter_Yaml, circuit_summary
from Circ_Plotter import Circ_Plotter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='data',help="input for everything")

    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-b','--backend',default='loc',
                        choices=['loc','ibm_s','ibm_q5','ibm_q16'],
                         help="backand for computations" )
    parser.add_argument('-s','--shots',type=int,default=2002, help="shots")
    parser.add_argument( "-G","--plotGates", action='store_true', default=False,
                         help="print full QASM gate list ")


    args = parser.parse_args()
    args.prjName='yieldDeconvol'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



#- - - - - - - - - - - -
def save_jobResult_Yaml(job_result,outF):
    print(type(job_result))
    #assert type(job_result) == type(qiskit.result.result.Result)
    outD=job_result.to_dict()
    print('save_jobResult_Yaml keys:',outD.keys())
    write_yaml(outD,outF)
    
#- - - - - - - - - - - -
def restore_jobResult_Yaml(results,outF): # needs a stub results object
    blob=read_yaml(outF)
    #print('old counts:',results.get_counts())
    results=results.from_dict(blob)
    #print('new counts:',results.get_counts())
    return results
    

   
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
plot=Circ_Plotter(args )

# Import Qiskit classes here - I want to controll matlpot lib from Circ_Plotter(.)
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

# fix it
#IBMQ.load_accounts()

# - - - - - - - - - - - - - - - -
print('\n\n  save/restore  jobResult \n')
# - - - - - - - - - - - - - - - -

# Make a 3Q GHZ state on 5 qubit register
nqub=5; nbit=3
qr = qiskit.QuantumRegister(nqub)
cr = ClassicalRegister(nbit)
ghz = QuantumCircuit(qr, cr)
ghz.h(qr[2])
ghz.cx(qr[2], qr[3])
ghz.cx(qr[3], qr[4])
ghz.measure(qr[2],cr[0])
ghz.measure(qr[3],cr[1])
ghz.measure(qr[4],cr[2])
print('ideal GHZ',ghz)

backend = qiskit.Aer.get_backend('qasm_simulator')
job = qiskit.execute([ghz], backend=backend, shots=args.shots)
results = job.result()
ideal_counts = results.get_counts()
print('M:ghz ideal counts:',ideal_counts)

# save job results to file
outF=args.outPath+'/jobRes55.yaml'
save_jobResult_Yaml(results,outF)

# to restore I need to provide valid jobResult object
circ0 = QuantumCircuit(1, 1)
circ0.measure(0,0)
job = qiskit.execute(circ0, backend=backend, shots=1)
results = job.result()

results2=restore_jobResult_Yaml(results,outF)
print('M: restored counts:', results2.get_counts())


# - - - - - - - - - - - - - - - -
print('\n\n     save/restore  CompleteMeasFitter()  \n')
# - - - - - - - - - - - - - - - -


''' Assume that we would like to generate a calibration matrix for the 3 qubits Q2, Q3 and Q4 in a 5-qubit Quantum Register [Q0,Q1,Q2,Q3,Q4].
Since we have 3 qubits, there are $2^3=8$ possible quantum states.
'''
from qiskit.providers.aer import noise
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

# Generate the calibration circuits
qubit_list = [2,3,4]
fake_nqubit_hw=max(qubit_list)+1
meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='jan1cal')

# output are 2 lists
print('state_labels=',state_labels)
print('meas_calibs len=',len(meas_calibs))

iexp=5
circ1=meas_calibs[iexp]
print('example circuit iexp=%d\n'%iexp,circ1)

q1_ReadErr=[[0.9, 0.1],[0.25,0.75]]
print('\n - - - - - - - - - - -  prepare  noise model:',q1_ReadErr)
# Generate a noise model for the 5 qubits
noise_model = noise.NoiseModel()

for qi in qubit_list: 
    #if qi !=4: continue
    read_err = noise.errors.readout_error.ReadoutError(q1_ReadErr)
    print('qi=',qi,read_err)
    noise_model.add_readout_error(read_err, [qi])

print('noise_model=',noise_model)

# 
print('\n Computing the Calibration Matrix w/o noise --> will be diagonal')

# Execute the calibration circuits without noise
backend = qiskit.Aer.get_backend('qasm_simulator')

job = qiskit.execute(meas_calibs, backend=backend, shots=args.shots, noise_model=noise_model)
cal_results = job.result()
# The calibration matrix with noise:
meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='jan1cal')

#print_measFitter(meas_fitter)

# save job results to file
outD={ 'calibInfo':{'qubit_list':qubit_list, 'nqubit_hw':fake_nqubit_hw}}
outF=args.outPath+'/measFitter.yaml'
save_measFitter_Yaml(meas_fitter,outF, outD)

print('M: before save:')
#?print_measFitter(meas_fitter)

meas_fitter3, calibMetaD=restore_measFitter_Yaml(outF)
print('M: after restore:')
#?print_measFitter(meas_fitter3)

print(' verify the restored measFilter is correct')
print('ideal GHZ circuit'); print(ghz)

job = qiskit.execute([ghz], backend=backend, shots=args.shots)
results = job.result()
# Results without mitigation
ideal_counts = results.get_counts()
print('ghz ideal counts:',ideal_counts)

job = qiskit.execute([ghz], backend=backend, shots=args.shots, noise_model=noise_model)
results = job.result()
# Results without mitigation
raw_counts = results.get_counts()
print('ghz         raw counts: 000=%.1f, 111=%.1f,  sum2=%.1f of shots=%d'%(raw_counts['000'],raw_counts['111'],raw_counts['000']+raw_counts['111'],args.shots ))

# Get the oryginal filter object
measFilter = meas_fitter.filter
mitig1_results = measFilter.apply(results)
mitig1_counts = mitig1_results.get_counts(0)
print('ghz mitig1 counts: 000=%.1f, 111=%.1f,  sum2=%.1f of shots=%d'%(mitig1_counts['000'],mitig1_counts['111'], mitig1_counts['000']+mitig1_counts['111'],args.shots ))

# Get restored filter3 object
measFilter3 = meas_fitter3.filter
mitig3_results = measFilter.apply(results)
mitig3_counts = mitig3_results.get_counts(0)
print('ghz mitig3 counts: 000=%.1f, 111=%.1f,  sum2=%.1f of shots=%d'%(mitig3_counts['000'],mitig3_counts['111'], mitig3_counts['000']+mitig3_counts['111'],args.shots ))



# - - - - - - - - - - - - - - - -
print('\n\n     save/restore  circuit as qasm   \n')
# - - - - - - - - - - - - - - - -
# https://qiskit.org/documentation/api/qiskit.circuit.QuantumCircuit.html

circF=args.dataPath+'/hidStr_2q.qasm'
circOrg=QuantumCircuit.from_qasm_file( circF )
print('\ncirc original fname: ', circF); print(circOrg,'\n')
outD=circuit_summary(circOrg)


#  convert gates to native
circBasis = circOrg.decompose()
print('\ncirc decomposed: '); print(circBasis,'\n')
circStr=circBasis.qasm()
circF2=args.outPath+'/inBasis.qasm'
with open(circF2,'w') as fd:  
    fd.write(circStr)
print('saved new circ to',circF2)



print('J-end')
