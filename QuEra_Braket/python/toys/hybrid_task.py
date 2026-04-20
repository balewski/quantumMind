#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from braket.aws import AwsDevice
from braket.tracking import Tracker
from braket.devices import LocalSimulator
from braket.ahs import AnalogHamiltonianSimulation, AtomArrangement, DrivingField, Hamiltonian

import os,time,json
import numpy as np
from pprint import pprint
from decimal import Decimal
#from myFunc import myFunc  # should exist on persistent S3

''' summary of relevant strings in SM_TRAINING_ENV 

 'channel_input_dirs': {'input': '/opt/ml/input/data/input'},
 'input_config_dir': '/opt/ml/input/config',

 'hyperparameters': {'backend_name': 'aquila',
                     'deviceArn': 'arn:aws:braket:us-east-1::device/qpu/quera/Aquila',
                     'num_iter': 5},
 'input_dir': '/opt/ml/input',
 'job_name': '12732784-434b-4127-9bd4-54edd5606d38',
 'model_dir': '/opt/ml/model',
 'module_dir': '/opt/ml/code',
 'output_data_dir': '/opt/ml/output/data',
 'output_dir': '/opt/ml/output',
 'user_entry_point': 'braket_container.py'}
'resource_config': {'current_group_name': 'homogeneousCluster',
                     'current_host': 'algo-1',
                     'current_instance_type': 'ml.m5.large',
                     'hosts': ['algo-1'],
                     'instance_groups': [{'hosts': ['algo-1'],
                                          'instance_group_name': 'homogeneousCluster',
                                          'instance_type': 'ml.m5.large'}],
                     'network_interface_name': 'eth0'},

SM_USER_ARGS=["--backend_name","aquila","--deviceArn","arn:aws:braket:us-east-1::device/qpu/quera/Aquila","--num_iter","5"]

'''

#...!...!....................
def inspect_dir(head,text='my123'):
    print('\n----Inspct %s  head=%s'%(text,head))   
    directory_contents = os.listdir(head)
    # Print the list of contents
    for item in directory_contents:
        c='/' if os.path.isdir(os.path.join(head,item)) else ''        
        print(item+c)

#...!...!....................
def nicer_env_vars(name):
    tmpJ=os.environ[name]
    tmpD=json.loads(tmpJ)
    print('\n----- %s dump:'%name)
    pprint(tmpD)
    return tmpD
        
#...!...!....................
def ahs_problem():
    a = 6.1e-6  # meters
    N_atoms = 11

    register = AtomArrangement()
    for i in range(N_atoms):   register.add([0.0, i*a])

    print('\nAHSP:dump atom coordinates')
    
    xL=register.coordinate_list(0)
    yL=register.coordinate_list(1)
    for x,y in zip(xL,yL):  print(x,y)

    H = Hamiltonian()
    
    omega_max = 2e6 * 2 * np.pi
    detuning_min = -9e6 * 2 * np.pi
    detuning_max = 7e6 * 2 * np.pi
    
    time_max = Decimal('4')/1000000 # 4 us
    time_ramp =time_max/4

    # rounding is required for Aquila
    omega_max=int(omega_max/400)*400
    detuning_min=int(detuning_min*5)/5
    detuning_max=int(detuning_max*5)/5
    
    time_points = [0, time_ramp, time_max - time_ramp, time_max]
    omega_values = [0, omega_max, omega_max, 0]
    detuning_values = [detuning_min, detuning_min, detuning_max, detuning_max]
    phase_values = [0, 0, 0, 0]

    drive = DrivingField.from_lists(time_points, omega_values, detuning_values, phase_values)
    H += drive

    print('\nAHSP:dump drive Amplitude :\n',drive.amplitude.time_series.times(), drive.amplitude.time_series.values())

    program = AnalogHamiltonianSimulation(  register=register, hamiltonian=H)

    return program

#...!...!....................
def execute_ahs_problem(device,ahs_prog,shots,verb=1):
    task = device.run(ahs_prog, shots=shots)
    results=task.result()  # this forces wait for task completion/termination, cost tracking is not working
    print('EAP: task status=%s:'%task.state())

    rawBitstr= None
    if task.state()=='COMPLETED': 
        rawBitstr=results.get_counts()
        numSol=len(rawBitstr)
        print('COMPLETED task,  numSol=%d'%numSol)
        if verb>1:
            print('COMPLETED, rawBitstr:');  pprint(rawBitstr)
    return task.state(), rawBitstr
                    
        
#...!...!....................
def run_hybrid_task(num_iter=2, deviceArn='fixMe', shots=100, backend_name='emul'):
    num_iter=int(num_iter)
    print('START hybrid task, num_iter=%d  deviceArn=%s backN=%s'%(num_iter,deviceArn,backend_name))
        
    # Use the device declared in the job script
    if backend_name=='aquila':
        assert 'Aquila' in deviceArn        
        device = AwsDevice(deviceArn)
        print('RHS: device is Aquila')
    else:
        from braket.devices import LocalSimulator
        device = LocalSimulator("braket_ahs")
        print('RHS: device is braket_ahs')

    envD=nicer_env_vars('SM_TRAINING_ENV')
    hpar=envD['hyperparameters']
    #.... save config as JSON
    outF='/opt/ml/output/data/ahs_task_conf.%s.json'%(backend_name)
    with open(outF, 'w') as json_file:
        md={'name':'aaa123', 'num_iter':num_iter}
        md['hyperparameters']= hpar
        json.dump(md, json_file)
    print('RHS: saved '+outF)
    outF+='2'
    with open(outF, 'w') as json_file:
        md={'name':'aaa123', 'num_iter':num_iter}
        json.dump(md, json_file)
    

    # ... inspect directories ....
    inspect_dir(os.getcwd(), 'working dir')
    inpPath='/opt/ml/input/data/input'
    inspect_dir(inpPath,'inpPath')
    outPath='/opt/ml/output/data'
    inspect_dir(outPath,'outPath')

    codePath=envD['module_dir']
    inspect_dir(codePath,'codePath')

    codePath+='/customer_code'
    inspect_dir(codePath,'customerCode')

    inspect_dir(codePath+'/original','customer-orig')
    inspect_dir(codePath+'/extracted','customer-extrct')
    exit(0)
   

    # test-read few lines form a file on S3
    file_name = "/opt/ml/input/data/input/HelloBraket_v1.py"

    print('\n Open the file for reading:',file_name)
    with open(file_name, "r") as file:
        # Read and print the first 3 lines
        for _ in range(3):
            line = file.readline().strip()
            if line:
                print(line)
    exit(0)
    # for now skip the rest....
    ahs_prog=ahs_problem()

    t0=time.time()
    for k in range(num_iter):
        print('\nRHS:executing %d task,   shots=%d...'%(k,shots))
        t1=time.time()
        status,rawBitstrD=execute_ahs_problem(device,ahs_prog,shots)
        t2=time.time()
        elaT=(t2-t0)/60
        if status!='COMPLETED':
            print('RHS: ABORTED task %d duration=%.1f (sec) elaT=%.1f (min), status=%s'%(k, t2-t1, elaT,status))
            exit(1)

        numSol=len(rawBitstrD)
        print('RHS: task %d done, numSol=%d, duration=%.1f (sec) elaT=%.1f (min), status=%s'%(k, numSol,t2-t1, elaT,status))
        if rawBitstrD!=None: 
            print('dump rawBitstrD:');  pprint(rawBitstrD)

        #.... save bitstrings as JSON
        outF='ahs_task%d.%s.json'%(k,backend_name)
        with open(outF, 'w') as json_file:
            json.dump(rawBitstrD, json_file)
        print('RHS: saved '+outF)
            
    print("RHS:M: %d tasks completed, elaT=%.1f (min)"%(num_iter,elaT))



#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
   
    run_hybrid_task(num_iter=3)
