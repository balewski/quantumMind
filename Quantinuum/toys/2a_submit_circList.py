#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
testing submitting a list of jcircuits to Quantinuum as 1 job
'''

import os,hashlib
import qnexus as qnx
from pytket import Circuit
#from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pprint import pprint
from time import time
     
#=================================
#  M A I N 
#=================================
if __name__ == "__main__":
    myTag='_'+hashlib.md5(os.urandom(32)).hexdigest()[:6]
    shots=50
    #devName="H1-Emulator"  # produces 0-cost
    devName="H1-1E"   # gives realsit cost estimate
    #myAccount='CSC641'
    myAccount='CHM170'  
    #project = qnx.projects.get_or_create(name="test-feb-14c")
    project = qnx.projects.get_or_create(name="qcrank-feb-14b")
    qnx.context.set_active_project(project)
    
    
    print('upload Bell-state circs ...')
    qc1 = Circuit(2).H(0).CX(0,1).measure_all()
    qc2 = Circuit(5).H(1).CX(1,2).measure_all()
    #print(tk_to_qiskit(qc2))
    
    #... upload 2 circuits
    t0=time()
    ref1 = qnx.circuits.upload(circuit=qc1, name='myca'+myTag)
    ref2 = qnx.circuits.upload(circuit=qc2, name='mycb'+myTag)
    t1=time()
    print('elaT=%.1f sec, uploaded, compiling ...'%(t1-t0))


    '''
    print('tt',type(ref1))
    xx=ref1.model_dump_json()
    #xx=ref1.json
    print('xx',type(xx))
    print(xx)
    '''

    devConf1 = qnx.QuantinuumConfig(device_name=devName,user_group=myAccount) # Noise-modelled emulator for Quantinuum’s H1 device, hosted in the cloud. gives 0-cost
    #devConf1 = qnx.QuantinuumConfig(device_name="H1-1E")  # Noise-modelled emulator for Quantinuum’s H1 device, hosted on dedicated hardware.  Gives cost, but will not execute

    print('use devConf:',devConf1)
    
    #...  compile list of circs at once
    t0=time()
    refCL=qnx.compile( circuits=[ref1,ref2], name='comp'+myTag,
                       optimisation_level=2, backend_config=devConf1,
                       project=project )
    t1=time()
    print('elaT=%.1f sec, compiled, executing ...'%(t1-t0))
    #... get cost
    shotL=[shots,shots*2]
    nCirc=len(refCL)    
    for ic in range(nCirc):
        cost=qnx.circuits.cost( circuit_ref=refCL[ic],n_shots=shots,
                                backend_config=devConf1, syntax_checker="H1-1SC"  )
        print('is=%d shots=%d cost=%.1f:'%(ic,shotL[ic],cost))
 
    #.... execution     
    t0=time()
    ref_exec= qnx.start_execute_job( circuits=refCL, n_shots=shotL,
                                     backend_config=devConf1,name="exec"+myTag)    
    t1=time()
    qnx.jobs.wait_for(ref_exec)
    results = qnx.jobs.results(ref_exec)
    t2=time()
    print('job submit elaT=%.1f,  execution elaT=%.1f\n'%(t1-t0,t2-t1))
    
    for ic in range(nCirc):
        result = results[ic].download_result()
        print('\nis=%d shots=%d res:'%(ic,shotL[ic])); pprint(result.get_counts())
    print('\n done devName=',devName)

    print('status2:',qnx.jobs.status(ref_exec))
