#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Correct Measurement Noise on a 3Q GHZ State
 Real HW only, matches SMAP qubits to circuit qubits, measures both at once
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
from Circ_Util import access_backend
from qiskit.converters import circuit_to_dag
from qiskit.tools.monitor import job_monitor
import qiskit as qk

# Import measurement calibration functions
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter)

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    #parser.add_argument('-o',"--outPath",default='out',help="output path for plots")

    parser.add_argument('-b','--backName',default='ibmq_belem',help="backand for computations" )
    parser.add_argument('-q','--numQubit',type=int,default=3, help="num phys qubits")
    parser.add_argument('-n','--numShots',type=int,default=4002, help="shots")
    parser.add_argument('-L','--optimLevel',type=int,default=3, help="transpiler ")
    parser.add_argument( "-E","--executeCircuit", action='store_true', default=False, help="may take long time and disable X-term")


    args = parser.parse_args()
    args.rnd_seed=111 # for transpiler

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()

    print('\n\n ******  NEW circuit : GHZ  ****** ')
    nq=args.numQubit
    ghz = qk.QuantumCircuit(nq, nq,name='ghz')
    ghz.h(0)
    for idx in range(1,nq):
        ghz.cx(0,idx)
    ghz.barrier(range(nq))
    ghz.measure(range(nq), range(nq))
    print(ghz)
    print('GHZ Depth:', ghz.depth())
    print('Gate counts:', ghz.count_ops())

    #------ verify backend is sensible ----
    backend=access_backend(args.backName)
    print('base gates:', backend.configuration().basis_gates)
    assert not backend.configuration().simulator

    print(' Layout using optimization_level=',args.optimLevel)

    circT = qk.transpile(ghz, backend=backend, optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed)
    tit='optimLevel=%d, %s'%(args.optimLevel,backend)
    print(circT)
    print('circT Depth:', circT.depth(), tit)
    print('Gate counts:', circT.count_ops())

    print('get dag for circT:')
    # see for explanation: https://bitbucket.org/balewski/quantummind/src/master/Qiskit/toy/circ2cyclePrint.py 
    qdag = circuit_to_dag(circT)
    pprint(qdag.properties())
    *_,lay=qdag.layers()  # get last item of iterator
    phys_qid=[]
    for op in lay['graph'].op_nodes():
        assert op.name=='measure'
        assert len(op.qargs)==1
        qL=[qub._index for qub in  op.qargs]
        phys_qid.append(qL[0])
        #print('  ',op.name, 'q'+str(qL))
    numChipQubits=circT.num_qubits
    print('\nM: circT phys qubits:',phys_qid,' backend=%s has %d qubits'%(args.backName,numChipQubits))

    # assemble circuits for SPAM using the same registeres in the same order

    # Generate the calibration circuits
    qr = qk.QuantumRegister(numChipQubits)
    qubit_list = phys_qid
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

    # output are 2 lists
    print('SPAM state_labels=',state_labels)
    print('meas_calibs len=',len(meas_calibs))

    iexp=5
    circ1=meas_calibs[iexp]
    print('example SPAM circuit iexp=%d\n'%iexp,circ1)


    # trick1: add ghz circ to the SPAM circuits to have just one execution on the HW
    meas_calibs.append(circT)
    print('GHZ added for SPAM circuits, num circ=',len(meas_calibs))

    if not args.executeCircuit:
        print('NO execution of circuit')
        exit(0)

    # ----- submission ----------
    job =  backend.run(meas_calibs,shots=args.numShots)
    jid=job.job_id()

    print('submitted JID=',jid,backend ,'\n now wait for execution of your circuit ...')
    job_monitor(job)

    cal_results = job.result()
    # Calculate the calibration matrix with the noise model
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')

    filtM=meas_fitter.cal_matrix
    print('meas_fitter shape:',filtM.shape,'shots=',args.numShots)
    print('\nprep state j: ',end='')
    [ print('%3d     '%j,end='') for j in range(filtM.shape[0]) ]
    for i in range(filtM.shape[0]):
        print('\nmeasuerd i=%2d'%i,end='')
        [ print('%6.3f  '%filtM[i,j],end='') for j in range(filtM.shape[0]) ]

    print('\n det M=%.3g %s  phys_qid:%s'%(np.linalg.det(filtM),args.backName,str(phys_qid)))

    print('\n  - - - - - - - - - - -  Analyzing the meas-err Results ')

    print(' qubit_list',list(meas_fitter.qubit_list))

    # What is the measurement fidelity?
    print("Average Measurement Fidelity: %f" % meas_fitter.readout_fidelity())

    # What is the measurement fidelity of soft-Q0?
    if nq==3: print("Average Measurement Fidelity of soft-Q0: %f" % meas_fitter.readout_fidelity(
        label_list = [['000','001','010','011'],['100','101','110','111']]))

    
    print('\ninspectmatrix , sum rows (axis=0), next columns')

    print(np.sum(filtM,axis=1))
    print(np.sum(filtM,axis=0))

    print('build matrix from raw yields used above, divide by shots')
    rawRes=job.result().get_counts()
    print('raw counts for %d circ'%len(rawRes))

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

    print('\n - - - -  Applying  SPAM-calib to  %dQ GHZ State on %s'%(NQ,args.backName))

    # GMZ results without mitigation
    results=cal_results
    raw_counts = results.get_counts(circT)
    [lab0,lab1]=[format(i, "0%db"%NQ) for i in [0,NB-1] ]
    #print('aaa',lab0,lab1)

    print('ghz         raw counts: %s=%.1f, %s=%.1f,  sum2=%.1f of shots=%d'%(lab0,raw_counts[lab0],lab1,raw_counts[lab1],raw_counts[lab0]+raw_counts[lab1],args.numShots ))
    # Get the filter object
    meas_filter = meas_fitter.filter

    # Results with mitigation
    mit_results = meas_filter.apply(results)
    mit_counts = mit_results.get_counts(circT)

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

    print('M:done',args.backName)

