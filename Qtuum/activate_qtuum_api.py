#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
testing access to Quantinuum API

*) for better security, you should retrieve your username and password from a secured storage.
*) available_devices() and and device_state() are the only two methods that require special handling because they are class methods. 

'''
import os
import qnexus as qnx
from pytket import Circuit
from pprint import pprint
from datetime import datetime
from pathlib import Path
from qnexus.client import get_nexus_client


#...!...!....................
def activate_qtuum_api():
    import os
    import qnexus as qnx
    MY_QTUUM_NAME=os.environ.get('MY_QTUUM_NAME')
    MY_QTUUM_PASS=os.environ.get('MY_QTUUM_PASS')
    print('credentials MY_QTUUM_NAME=',MY_QTUUM_NAME,MY_QTUUM_PASS)
    qnx.auth._request_tokens(MY_QTUUM_NAME,MY_QTUUM_PASS)
    # List your saved credentials
    my_credentials = qnx.credentials.get_all()
    print(my_credentials)  # it prints alwasy empty list

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    #my_credentials = qnx.credentials.get_all()
    #print(my_credentials)
    
    if 1: # works
        activate_qtuum_api()
        #exit(0)

    if 0: # works too, but need manual typing 
        qnx.login_with_credentials()        
        my_credentials = qnx.credentials.get_all()
        pprint(my_credentials)
        exit(0)
        
    if 0: # List Quantinuum devices+calibration  ~2 pages of text
        xxL=qnx.devices.get_all( issuers=[qnx.devices.IssuerEnum.QUANTINUUM] )
        for x in xxL: print('\n',x)
        ttt
        
    project = qnx.projects.get_or_create(name="test-feb-13")
    qnx.context.set_active_project(project)
   
    dateTag = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
    print('define Bell-state circ')
    qc1 = Circuit(2).H(0).CX(0,1).measure_all()
    qc2 = Circuit(5).H(1).CX(0,1).measure_all()
    
    print('circ1 as commands:\n',qc1.get_commands())
    ref1 = qnx.circuits.upload(circuit=qc1, name="jan-circ4a_"+dateTag)
    print('\nupload ref1:',type(ref1))
    pprint(ref1)
    ref2 = qnx.circuits.upload(circuit=qc2, name="jan-circ4b_"+dateTag)

    config = qnx.QuantinuumConfig(device_name="H1-Emulator") # Noise-modelled emulator for Quantinuum’s H1 device, hosted in the cloud.
    #config = qnx.QuantinuumConfig(device_name="H2-Emulator") 
    #config = qnx.QuantinuumConfig(device_name="H1-1E")  # Noise-modelled emulator for Quantinuum’s H1 device, hosted on dedicated hardware.
    #config = qnx.QuantinuumConfig(device_name="H1-1SC")  # syntax checker
    #config = qnx.QuantinuumConfig(device_name="H2-1SC")  # syntax checker
    #config = qnx.QuantinuumConfig(device_name="H1-1")  # QPU

    
    ref_compile_job = qnx.start_compile_job(
        circuits=[ref1,ref2],
        backend_config=config,
        optimisation_level=2,
        name="comp-job_"+dateTag
    )

    print('\nupload refT:')
    pprint(ref_compile_job)

    qnx.jobs.wait_for(ref_compile_job)
    ref_compiled_circuit = qnx.jobs.results(ref_compile_job)[0].get_output()
    print('\nupload refCC:')
    pprint(ref_compiled_circuit)

    print('submit to',config)
    ref_execute_job = qnx.start_execute_job(
        circuits=[ref_compiled_circuit],
        n_shots=[200],
        backend_config=config,
        name="exec-job_"+dateTag
    )

    print('\nstatus1:',qnx.jobs.status(ref_execute_job))
    
    qnx.jobs.wait_for(ref_execute_job)
    results = qnx.jobs.results(ref_execute_job)
    nCirc=len(results)
    for ic in range(nCirc):
        result = results[i].download_result()
        print('is=%d res:'%ic); pprint(result.get_counts())
    print(config)

    print('status2:',qnx.jobs.status(ref_execute_job))
    
    job_name = f"Job from {datetime.now()}"
    qnx.filesystem.save(
        ref=ref_execute_job,
        path=Path.cwd() / "my_job_folder" / job_name,
        mkdir=True,
    )
    my_job_ref = qnx.filesystem.load(
        path=Path.cwd() / "my_job_folder" / job_name
    )
    print('\nstatus3:',qnx.jobs.status(my_job_ref))

    exit(0)
    
   




