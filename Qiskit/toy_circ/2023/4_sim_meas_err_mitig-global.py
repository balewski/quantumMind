#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
The measurement calibration is used to mitigate measurement errors. 
Simulation only 
Measures SPAM for a subset of qubits on a chip

Update 2022-05
based on 
https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/ignis/measurement_error_mitigation.ipynb


'''
# Import general libraries (needed for functions)
import numpy as np
import time,os,sys
from pprint import pprint
import scipy.linalg as la
sys.path.append(os.path.abspath("../utils/"))
from Circ_Plotter import Circ_Plotter

#np.set_printoptions(precision=3)
#np.set_printoptions(suppress=True)

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-n','--numShots',type=int,default=4002, help="shots")
    parser.add_argument( "-G","--plotGates", action='store_true',
                         default=False,help="plot circuit")


    args = parser.parse_args()
    args.prjName='yieldDeconvol'
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":

    args=get_parser()
    plot=Circ_Plotter(args )

    # Import Qiskit classes here - I want to controll matlpot lib from Circ_Plotter(.)
    import qiskit as qk
    from qiskit.providers.aer import noise
    from qiskit.tools.visualization import plot_histogram

    # Import measurement calibration functions
    from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)


    ''' Assume that we would like to generate a calibration matrix for the 3 qubits Q2, Q3 and Q4 in a 5-qubit Quantum Register [Q0,Q1,Q2,Q3,Q4].
    Since we have 3 qubits, there are $2^3=8$ possible quantum states.
    '''

    # Generate the calibration circuits
    qr = qk.QuantumRegister(5)
    qubit_list = [2,4,3]
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

    # output are 2 lists
    print('SPAM state_labels=',state_labels)
    print('meas_calibs len=',len(meas_calibs))

    if 0:  # add extra circut for testing if SMPAM & target circ can be measured simultanously
        nq=3
        circ0 = qk.QuantumCircuit(nq, nq,name='jan-circ')
        circ0.h(0)
        for idx in range(1,nq):
            circ0.cx(0,idx)
            circ0.barrier(range(nq))
            circ0.measure(range(nq), range(nq))
        print('add to SPAM circ0:',circ0)
        meas_calibs.append(circ0)

    iexp=5
    circ1=meas_calibs[iexp]
    print('example SPAM circuit iexp=%d\n'%iexp,circ1)
    if args.plotGates: # it mast be called *after* other drawings
        plot.save_me( plot.circuit(circ1), 9)

    # 
    print('\n Computing the Calibration Matrix w/o noise --> will be diagonal')

    # https://qiskit.org/documentation/api/qiskit.ignis.mitigation.measurement.CompleteMeasFitter.html

    # Execute SPAM calibration circuits without noise
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(meas_calibs, backend=backend, shots=args.numShots)
    cal_results = job.result()
    # The calibration matrix without noise is the identity matrix
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    print('ideal meas_fitter=\n',meas_fitter.cal_matrix)
    #print(meas_fitter)
    # other getters: state_labels, qubit_list, filter

    #qa_ReadErr=[[0.9, 0.1],[0.02,0.98]]
    qa_ReadErr=[[0.9, 0.1],[0.25,0.75]]  # qa is very poor quality
    qb_ReadErr=[[0.95, 0.05],[0.01,0.99]]
    print('\n - - - - - - - - - - -  Computing the Calibration Matrix\n w/ per-qubit noise model')
    # Generate a noise model for the 3 qubits, Q0 w/o noise
    noise_model = noise.NoiseModel()

    for qi in qubit_list: 
        if qi ==3:  # qubit ID is 0...4 but bits are 0,1,2
            read_err = noise.errors.readout_error.ReadoutError(qb_ReadErr)
        else:
            read_err = noise.errors.readout_error.ReadoutError(qa_ReadErr)
        print('qi=',qi,read_err)
        noise_model.add_readout_error(read_err, [qi])

    print('\nnoise_model=',noise_model)

    # Execute SPAM calibration circuits w/ noise 10x more shots
    backend = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(meas_calibs, backend=backend, shots=args.numShots*10, noise_model=noise_model)
    cal_results = job.result()
    # Calculate the calibration matrix with the noise model
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')

    filtM=meas_fitter.cal_matrix
    print('non-trivial meas_fitter shape:',filtM.shape)
    print('\nprep state j:',end='')
    [ print('%3d     '%j,end='') for j in range(filtM.shape[0]) ]
    for i in range(filtM.shape[0]):
        print('\nmeasuerd i=%d'%i,end='')
        [ print('%6.3f  '%filtM[i,j],end='') for j in range(filtM.shape[0]) ]
    print('\n det M=%.3f %s  phys_qid:%s'%(np.linalg.det(filtM),backend,str(qubit_list)))

    
    print('\n  - - - - - - - - - - -  Analyzing the meas-err Results ')
    '''
    We would like to compute the total measurement fidelity, and the measurement fidelity for a specific qubit, for example, Q0.  Must separate bit strings to those w/ state0 or 1.

    Since the on-diagonal elements of the calibration matrix are the probabilities of measuring state 'x' given preparation of state 'x', then the trace of this matrix is the average assignment fidelity.
    '''

    print(' qubit_list',list(meas_fitter.qubit_list))

    # What is the measurement fidelity?
    print("Average Measurement Fidelity: %f" % meas_fitter.readout_fidelity())

    # What is the measurement fidelity of Q0? worsk only for 3 qubit circuit

    print("Average Measurement Fidelity of Q0: %f" % meas_fitter.readout_fidelity(
        label_list = [['000','001','010','011'],['100','101','110','111']]))


    print('\ninspectmatrix , sum rows (axis=0), next columns')

    print(np.sum(filtM,axis=1))
    print(np.sum(filtM,axis=0))

    print('build matrix from raw yields used above, divide by shots')
    rawRes=job.result().get_counts()
    #print('raw counts:'); pprint(rawRes)

    NQ=meas_calibs[0].num_clbits
    NB=1<<NQ  # num bit-strings
    print('M: num measured bits=%d -->%d bit-strings'%(NQ,NB))

    # re-pack raw counts as matrix
    # mappig of strings to 1-d index
    sdx={format(i, "0%db"%NQ):i for i in range(NB)}   
    print('M:sdx:',sdx)
 
    calYn=np.zeros((NB,NB))
    for idx1,rec in enumerate(rawRes):    
        if idx1>=NB : break  # there may be more circuits on the list
        for lab in rec:
            calYn[ sdx[lab],idx1]=rec[lab]/args.numShots

    print('calYn:\n',calYn)

    invM=la.inv(calYn)
    print('calYn inv:\n',invM)


    print('\n - - - -  Applying  SPAM-calib to  %dQ GHZ State on %s'%(NQ,backend))
    '''
    We now perform another experiment and correct the measured results fro SPAM

    Correct Measurement Noise on a 3Q GHZ State
    We start with the 3-qubit GHZ state on the qubits Q2,Q3,Q4:
    '''
    # Make a 3Q GHZ state
    cr = qk.ClassicalRegister(3)
    ghz = qk.QuantumCircuit(qr, cr)  # use the same qr as for spam
    ghz.h(qr[2])  # use the same qubits as for SPAM
    ghz.cx(qr[2], qr[3])
    ghz.cx(qr[3], qr[4])
    ghz.measure(qr[2],cr[0])  # set order to match SPAM measurement qubit list!
    ghz.measure(qr[3],cr[2])  # <== HERE
    ghz.measure(qr[4],cr[1])
    print('ideal GHZ',ghz)

    job = qk.execute([ghz], backend=backend, shots=args.numShots)
    results = job.result()
    # Results without mitigation
    ideal_counts = results.get_counts()
    print('ghz ideal counts:',ideal_counts)

    job = qk.execute([ghz], backend=backend, shots=args.numShots, noise_model=noise_model)
    results = job.result()
    # Results without mitigation
    raw_counts = results.get_counts()
    [lab0,lab1]=[format(i, "0%db"%NQ) for i in [0,NB-1] ]
    #print('aaa',lab0,lab1)

    print('ghz         raw counts: %s=%.1f, %s=%.1f,  sum2=%.1f of shots=%d'%(lab0,raw_counts[lab0],lab1,raw_counts[lab1],raw_counts[lab0]+raw_counts[lab1],args.numShots ))

    # Get the filter object
    meas_filter = meas_fitter.filter

    # Results with mitigation
    mit_results = meas_filter.apply(results)
    mit_counts = mit_results.get_counts(ghz)
                 
    print('ghz mitigated counts: %s=%.1f, %s=%.1f,  sum2=%.1f of shots=%d'%(lab0,mit_counts[lab0],lab1,mit_counts[lab1], mit_counts[lab0]+mit_counts[lab1],args.numShots ))

    # manual mitigation of read-error using inverted rawY matrix
    rawY=np.zeros(NB) 
    for lab in raw_counts:
            rawY[ sdx[lab]]=raw_counts[lab]
    #print('rawY:\n',rawY)
    ymm=np.matmul(invM,rawY)
    #print('ymm:\n',ymm)

    print(mit_counts)
    print('label, ghz noisy , mitigate counts, matInv counts:')
    for x in raw_counts:
        ym='none'
        if x in mit_counts:
            ym='%.1f'%mit_counts[x]
        print('%s %d  %s  %.1f'%(x, raw_counts[x],ym, ymm[sdx[x]]))


    ''' Further examples

    1) 
    Applying to a reduced subset of qubits
    Consider now that we want to correct a 2Q Bell state, but we have the 3Q calibration matrix. We can reduce the matrix and build a new mitigation object

    2) 
    Tensored mitigation
    The calibration can be simplified if the error is known to be local. By "local error" we mean that the error can be tensored to subsets of qubits. In this case, less than $2^n$ states are needed for the computation of the calibration matrix.

    '''


    # Plot the calibration matrix
    # self._tens_fitt.plot_calibration(0, ax, show_plot)
    ax=plot.blank_page(13)
    meas_fitter.plot_calibration(ax=ax,show_plot=False)
    tit='qubits: '+''.join(str(qubit_list))
    tit2=backend.name()
    tit3='avr meas fidel: %.3f'%meas_fitter.readout_fidelity()
    ax.text(0.07,0.15,tit3,color='b',transform=ax.transAxes)
    ax.text(0.07,0.1,tit,color='b',transform=ax.transAxes)
    ax.text(0.07,0.05,tit2,color='b',transform=ax.transAxes)


    # We can now plot the results with and without error mitigation:
    fig=plot_histogram([raw_counts, mit_counts,ideal_counts], legend=['raw', 'mitigated','ideal'])
    plot.save_me(fig, 14)


    plot.display_all()

