#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
  Fit U3 to a random 1-qubit state , using loss  from Hadamard-overlap test
 Input:
  |a>   1-qubit state from parametrized U3
  |b> is  U3 random 1-qubit state  (target)
  
 Output: 
   theta, phi

use state vector

'''
import sys,os
from pprint import pprint
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.quantum_info import random_statevector, Statevector
import numpy as np
from scipy.optimize import minimize

#...!...!..................
def build_circ(par,op,type='re'):
    m=1
    qc = QuantumCircuit(2*m+1)  # add ancilla as qubit 0
    qc.h(0)
    if type=='im': qc.sdg(0)  # will compute imaginary component
    qc.u(par[0],par[1],par[2],1)
    qc.u(parT[0],parT[1],parT[2],2)
    for i in range(2):
        tgt=i+1
        if op=='x': qc.cx(0,tgt)
        elif op=='y': qc.cy(0,tgt)
        elif op=='z': qc.cz(0,tgt)
        else:  continue
    qc.h(0)
    if verb: print(qc)
    return qc 


#...!...!..................
def cost_func(parVal):
    global iterCount
    over=np.zeros(3)
    ops='xyz'
    
    iterCount+=1
    for j,op in enumerate(ops):
        slab='ov'+op  # state label
        qc=build_circ(parVal,op)
        qc.save_statevector(slab)
        result = backend.run(qc).result() 
        ov2 = result.data(0)[slab]
        if verb>1: print('out vec ',slab, ov2)
        p0,p1=ov2.probabilities([0])
        delp=p0-p1
        over[j]=delp
        if verb>0: print(slab,' p0,p1: %.3f %.3f '%(p0,p1), 'p0-p1: %.3f'%delp)
    l1=np.linalg.norm(over,ord=1)
    l2=np.linalg.norm(over)
    chi2=np.sum(over*over)
    if iterCount<10 or iterCount%10==0:
        print('%d step l1=%.3f l2=%.3f chi2=%.3f   '%(iterCount,l1,l2,chi2),end='')
        for i in range(3): print('d%s=%.3f, '%(ops[i],over[i]),end='')
        print('')

    return 1-chi2

#...!...!..................
def get_state(par):
    
    qc1 = QuantumCircuit(1)
    qc1.u(par[0],par[1],par[2],0)
    qc1.save_statevector('os')
    result = backend.run(qc1).result() 
    os = result.data(0)['os']
    print('out vec ',os.data)
    return os

def get_spherical_coordinates(statevector):
    # Convert to polar form:
    r0 = np.abs(statevector[0])
    ϕ0 = np.angle(statevector[0])

    r1 = np.abs(statevector[1])
    ϕ1 = np.angle(statevector[1])

    # Calculate the coordinates:
    r = np.sqrt(r0 ** 2 + r1 ** 2)
    θ = 2 * np.arccos(r0 / r)
    ϕ = ϕ1 - ϕ0
    return [r, θ, ϕ]
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":

    verb=1
    nPar=3

    parT=(0.5 -np.random.rand(nPar)) *np.pi/5  # define the range of variation 
   
    parVal0=np.random.rand(nPar)*np.pi/2
    backend = Aer.get_backend('aer_simulator')
   
    if 0: # testing components
        qc=build_circ(parVal0,'x')
        cost_func(parVal0)
        end1


    # the main optimizer loop
    numIter=100
    iterCount=0    
    verb=0
    out = minimize(cost_func, x0=parVal0, 
                   method="COBYLA", options={'maxiter':numIter})
    print('\nM:end of minimizer:\n', out)
    parOpt=out['x']
    print('parOpt',parOpt)

    # Compare true and found quantum state
    osT=get_state(parT)
    osF=get_state(parOpt)
    osD=osT-osF
    print('osDiff ',osD.data)
    print('par true',parT)
    print('M:done ')

    print('project on Bloch sphere:  r, theta, phi')  
    print('bloch true:',get_spherical_coordinates(osT))
    print('bloch fit: ',get_spherical_coordinates(osF))
