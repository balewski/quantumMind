import numpy as np
from pprint import pprint
import time,  os

from qiskit import  QuantumRegister
# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,
                                                 CompleteMeasFitter, TensoredMeasFitter)

#...!...!....................
def print_measFitter(meas_fitter, mxDim=3):
    filtM=meas_fitter.cal_matrix
    print('meas_fitter shape:',filtM.shape)
    state_labels=meas_fitter.state_labels
    maxLab=2**mxDim
    print("Average Measurement Fidelity: %f" % meas_fitter.readout_fidelity(),',meas qubit_list',list(meas_fitter.qubit_list))

    if filtM.shape[0] >maxLab:
        print('only diagonal measFitter  due to large size=',filtM.shape[0])
        for i in range(filtM.shape[0]):
            print('i=%d state=%s  prob=%.3f'%(i,state_labels[i],filtM[i,i]))
        return 
    
    print('state labels:',end='')
    [ print('%6s  '%j,end='') for j in meas_fitter.state_labels  ]
    print('\nprep state j:',end='')
    [ print('%3d     '%j,end='') for j in range(filtM.shape[0]) ]
    for i in range(filtM.shape[0]):
        print('\nmeasuerd i=%d'%i,end='')
        [ print('%6.3f  '%filtM[i,j],end='') for j in range(filtM.shape[0]) ]
    print() 


#............................
#............................
#............................
class ReadErrCorrTool(object):
    def __init__(self, conf):
        self.conf=conf

#...!...!....................
    def make_circuits(self,verb=1):
        # Generate the calibration circuits
        qr = QuantumRegister(self.conf['nqubit'])
        
        circL, state_labels = complete_meas_cal(qubit_list=self.conf['meas_qubit'], qr=qr, circlabel=self.conf['cal_name'])

        if verb>0:
            iexp=1
            print('RECT: example circuit iexp=%d\n'%iexp,circL[iexp])

        print('state_labels=',state_labels)
        print('meas_calibs len=',len(circL))
        self.conf['state_labels']=state_labels
        circIdL=[ x.name  for x in circL]
        self.conf['circId']=circIdL
        print('circId :',circIdL)
        return circL
        

#...!...!....................
    def compute_corrections(self,job):
        cal_results = job.result()

        raw_counts = cal_results.get_counts(0)
        print('FF',raw_counts)
        
        # The calibration matrix without noise is the identity matrix
        meas_fitter = CompleteMeasFitter(cal_results,
                            qubit_list=self.conf['meas_qubit'],
                            state_labels=self.conf['state_labels'],
                            circlabel=self.conf['cal_name'])
        
        print_measFitter(meas_fitter, mxDim=3)
        
        self.meas_fitter=meas_fitter

        
#...!...!....................
    def apply_corrections(self,job):
        results=job.result()
        meas_filter = self.meas_fitter.filter
        startT = time.time()
        # Results with mitigation
        mitigated_results = meas_filter.apply(results)
        jobAllD = mitigated_results.to_dict()
        print('job err-mitig took %d sec'%(time.time()-startT))
        jobMResD0=jobAllD.pop('results')

        # retain non-calibration experiments
        jobMResL=[]
        for  exp in jobMResD0:
            if  self.conf['cal_name'] in exp['header']['name']: continue
            jobMResL.append(exp)
        return jobMResL
  
