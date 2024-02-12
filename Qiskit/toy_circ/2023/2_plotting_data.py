#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Exercise  different plottinig tools
based on  2_plotting_data_in_qiskit.ipynb
'''
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
    parser.add_argument( "-G","--plotGates", action='store_true', default=False,
                         help="print full QASM gate list ")


    args = parser.parse_args()
    args.prjName='Bell_v1'

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
plot=Circ_Plotter(args )

# Import Qiskit classes here - I want to controll matlpot lib from Circ_Plotter(.)
from qiskit import *
 
# fix it
#IBMQ.load_accounts()

print('\n Plot histogram of yields for Bell state')
# quantum circuit to make a Bell state 
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)

meas = QuantumCircuit(2, 2)
meas.measure([0,1], [0,1])

# execute the quantum circuit 
backend = BasicAer.get_backend('qasm_simulator') # the device to run on
circ = bell+meas
print('M:circ');print(circ)
result = execute(circ, backend, shots=args.shots).result()
countsD  = result.get_counts(circ)
print('raw counts',countsD)

probD=do_yield_stats(countsD)
print('probabilities:',probD)

plot.measurement(probD)
if args.plotGates: 
    plot.save_me( plot.circuit(circ), 9)
    print(circ)

# add missing:  plot_histogram(exp_counts, sort='hamming', target_string='1011')

print('\n --------- Plot State ')
'''
code: /home/balewski/anaconda3/lib/python3.6/site-packages/qiskit/visualization/state_visualization.py


'plot_state_city': 
        The standard view for quantum states where the real and imaginary (imag) parts of the state matrix are plotted like a city
'plot_state_qsphere': 
        The Qiskit unique view of a quantum state where the amplitude and phase of the state vector are plotted in a spherical ball. The amplitude is the thickness of the arrow and the phase is the color. For mixed states it will show different 'qsphere' for each component.
'plot_state_paulivec': 
        The representation of the state matrix using Pauli operators as the basis $\rho=\sum_{q=0}^{d^2-1}p_jP_j/d$
'plot_state_hinton': 
        Same as 'city' but with the size of the element represents the value of the matrix element.
'plot_bloch_multivector': 
        The projection of the quantum state onto the single qubit space and plotting on a bloch sphere.
'''

from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

# execute the quantum circuit 
backend = BasicAer.get_backend('statevector_simulator') # the device to run on
result = execute(bell, backend).result()
psi  = result.get_statevector(bell)

plot.save_me( plot_state_city(psi,'psi as city'),21)
plot.save_me( plot_state_hinton(psi,'psi as hinton'),22 )
plot.save_me( plot_state_paulivec(psi ,'psi as paulivec'),23 )

plot.display_all()
