#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Investigate 3-qubit GHZ measurement under
1-qubit noise in the form of Pauli matrix rotation
Results:
https://docs.google.com/document/d/1Y3BUtTLDe6P8Ik2aZws2veI1G-kN8CPpdUS8GFC_Mwk/edit?usp=sharing
'''

from qiskit import *
import time,os,sys
sys.path.append(os.path.abspath("../utils/"))

from Circ_Plotter import Circ_Plotter
from Circ_Util import do_yield_stats

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-b','--backend',default='loc',
                        choices=['loc','ibm_s','ibm_q5','ibm_q16'],
                         help="backand for computations" )
    parser.add_argument('-s','--shots',type=int,default=1024, help="shots")
    parser.add_argument( "-G","--printGates", action='store_true', default=False,
                         help="print full QASM gate list ")


    args = parser.parse_args()
    args.prjName='noisyGHZ'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

from qiskit.quantum_info.operators import Operator
import numpy as np
import scipy.stats

#- - - - - - - - - - - - -
# operator rotating 1 qubit by an angle N(0,theta_std) around random 3D axis
def one_qubit_noisy_rot(theta_std):
    # base 4 1-qubit opertaors
    e1q_op= Operator([[ 1,0],[0,1]])
    x1q_op= Operator([[ 0,1],[1,0]])
    y1q_op= Operator([[0.+0.j, 0.-1.j], [0.+1.j, 0.+0.j]])
    z1q_op= Operator([[1,0],[0,-1]])
    print('y1q_op=',y1q_op)
    
    # rotation angle:
    theta = scipy.stats.truncnorm.rvs( -3, 3 ) * theta_std

    #direction of rotation for 1 qubit
    vec  = np.random.randn( 3 )
    vec /= np.linalg.norm ( vec )
    print('vec=',vec, np.sqrt(np.sum(vec*vec)))
    dir_op= vec[0]* x1q_op +  vec[1]* y1q_op +  vec[2]* z1q_op
    print('dir_op=',dir_op)
    op=  np.cos(theta/2) *e1q_op - (0+1j)* np.sin(theta/2)*dir_op
    print('op=',op)
    return op

    
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
plot=Circ_Plotter(args )

# fix it
#IBMQ.load_accounts()

# ------- Create a Quantum Circuit acting on a quantum register of three qubits
nq=3; nb=3
circ = QuantumCircuit(nq)
circ.h(0) # Add a H gate on qubit 0, putting this qubit in superposition.

# add noise on 1 qubit
#circ.ry(0.2,1)

# add Unitary custom matrix
'''
# CNOT matrix operator with qubit-0 as control and qubit-1 as target
cx_op = Operator([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]])
circ.unitary(cx_op, [0, 2], label='cx_01')
'''

circ.barrier(range(nq))
theta_std=0.5
for q in range(nq):
    noise_op=one_qubit_noisy_rot(theta_std)
    circ.unitary(noise_op, [q], label='noise_q%d'%q)

circ.cx(0, 1) # now  qubits 0,1  in a Bell state.
circ.cx(0, 2) # now we have 3-qubit Bell state

meas = QuantumCircuit(nq, nb)
meas.barrier(range(nq))
meas.measure(range(nq), range(nb))
circm = circ+meas # The Qiskit circuit object supports composition

# execute the quantum circuit 
backend = BasicAer.get_backend('qasm_simulator') # the device to run on
result = execute(circm, backend, shots=args.shots).result()
countsD  = result.get_counts(circm)
print('raw counts',countsD)

probD=do_yield_stats(countsD)
print('probabilities:',probD)

plot.measurement(probD)
if args.printGates: # it mast be called *after* other drawings
    plot.save_me( plot.circuit(circm), 9)
    print(circm)
    plot.display_all()



from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

# execute the quantum circuit 
backend = BasicAer.get_backend('statevector_simulator') # the device to run on
result = execute(circ, backend).result()
psi  = result.get_statevector(circ)

plot.save_me( plot_state_city(psi,'psi as city'),21)
plot.save_me( plot_state_paulivec(psi ,'psi as paulivec',figsize=(14,3)),23 )

plot.display_all()

#/home/balewski/anaconda3/lib/python3.6/site-packages/qiskit/visualization/state_visualization.py
