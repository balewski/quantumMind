#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 2) Submit readout callibration job to a device

Required INPUT:
 --transpName : hidStr_2qm.ibmq_boeblingen-Q5+6
 --shots 2048
 --mock (optional)
 
Reads:
  data/transp.ghz_5qm.v10.yaml

Prints/write  out/subInfo-jNNNNN.yaml  - job submission summary, fixed name

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint

from qiskit import  transpile , assemble, QuantumRegister, execute, Aer

from NoiseStudy_Util  import submitInfo_2_yaml
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, print_measFitter

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path ")
    parser.add_argument( "--sim", action='store_true',
                         default=False, help="run on simulator at IBM")

    parser.add_argument('-t','--transpName', default='hidStr_2qm.ibmq_boeblingen-Q5+6',
                        help=" transpiled circuit to be executed")

    parser.add_argument( "--mock", action='store_true',
                         default=False, help="run on simulator instantly")
    parser.add_argument('-s','--shots',type=int,default=8096, help="shots")

    args = parser.parse_args()
    args.prjName='subMitg_'+args.transpName
    args.rnd_seed=123  # for transpiler
    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

    
#...!...!....................
def prep_simulated_readout_noise(calibCircL, state_labels,qubit_list,shots):
    from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
    from qiskit.providers.aer import noise
    
    q1_ReadErr=[[0.9, 0.1],[0.25,0.75]]
    print('\n prep_simulated_readout_noise(), 1qubit model:',q1_ReadErr)

    noise_model = noise.NoiseModel()

    for qi in qubit_list: 
        read_err = noise.errors.readout_error.ReadoutError(q1_ReadErr)
        print('qi=',qi,read_err)
        noise_model.add_readout_error(read_err, [qi])

    print('noise_model=',noise_model)

    # Execute the calibration circuits
    backend = Aer.get_backend('qasm_simulator')
    job = execute(calibCircL, backend=backend, shots=shots, noise_model=noise_model)
    cal_results = job.result()
    # Calculate the calibration matrix with the noise model
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='jan23')
    print_measFitter(meas_fitter)

#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import complete_meas_cal 

transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
blobD=read_yaml(transpF)
if args.verb>2:  print('M:  transp blob'); pprint(blobD)

circD=blobD['circOpt']
if args.verb>1:    print('M: circOpt data:'); pprint(circD)

nqubit=circD['info']['qubits']
qubit_list=circD['info']['meas_qubit']
backName=circD['backend']
if args.sim: backName='ibmq_qasm_simulator' # redicrect to simulator

# Generate the calibration circuits
qr = QuantumRegister(nqubit)
calibCircL, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='jan23')
# output are 2 lists
numStates=len(state_labels)
print('state_labels=',state_labels)

iexp=numStates//2
circ1=calibCircL[iexp]
print('example calib circuit iexp=%d of %d'%(iexp,numStates))
print(circ1)

if args.mock:
    print('mock execution on simulator w/ noisy readout')
    prep_simulated_readout_noise(calibCircL, state_labels, qubit_list,args.shots)
    print('no jobs submitted to device, quit')
    exit(33)
    
backend=access_backend(backName)
print('Execute @ backend=%s, numStates=%d shots/state=%d'%(backend,numStates,args.shots))

# no transpiler
qobj_circ = assemble(calibCircL, backend, shots=args.shots)
job_exe =  backend.run(qobj_circ)

outD=submitInfo_2_yaml(job_exe,args ,circD)
print('\n *) Submitted calib:    %s\n'%args.transpName,outD,'\n')

outF=args.dataPath+'/submInfo-%s.yaml'%outD['submInfo']['jid6']
write_yaml(outD,outF)

print('\nEND-OK')
   
