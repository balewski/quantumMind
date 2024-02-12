from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.ahs import AnalogHamiltonianSimulation, AtomArrangement, DrivingField, Hamiltonian

import os,time,json
import numpy as np
from pprint import pprint
from decimal import Decimal

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
def run_hybrid_task(args):    
    print('START hybrid task, numIter=%d  device_arn=%s backN=%s'%(args.numIter,args.device_arn,args.backend_short))

    # Aquila ARN is also avaliable via: os.environ["AMZN_BRAKET_DEVICE_ARN"]
    
    if args.awsHybrid :
        deviceArn=args.device_arn
        #print('compare ARN:',args.device_arn,deviceArn)
        device = AwsDevice(deviceArn)
        print('RHS: device is Aquila')
    else:
        from braket.devices import LocalSimulator
        device = LocalSimulator("braket_ahs")
        print('RHS: device is braket_ahs')
    
    ahs_prog=ahs_problem()
    shots=args.numShots
    t0=time.time()
    for k in range(args.numIter):
        print('\nRHS:executing %d task of %d,   shots=%d...'%(k,args.numIter,shots))
        t1=time.time()
        status,rawBitstrD=execute_ahs_problem(device,ahs_prog,shots)
        t2=time.time()
        elaT=(t2-t0)/60
        numSol=len(rawBitstrD)
        print('RHS: task %d done, numSol=%d, duration=%.1f (sec) elaT=%.1f (min), status=%s'%(k, numSol,t2-t1, elaT,status))
        if rawBitstrD!=None: 
            print('dump rawBitstrD:');  pprint(rawBitstrD)

        #.... save bitstrings as JSON
        outF=os.path.join(args.outPath,'ahs_task%d.%s.json'%(k,args.backend_short))
        with open(outF, 'w') as json_file:
            json.dump(rawBitstrD, json_file)
        print('RHS: saved '+outF)
            
    print("RHS:M: %d tasks completed, elaT=%.1f (min)"%(args.numIter,elaT))
