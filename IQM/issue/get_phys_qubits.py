#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


import sys,os,hashlib
import numpy as np
from pprint import pprint
from iqm.qiskit_iqm import IQMProvider, transpile_to_IQM
from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister

#...!...!....................
def create_ghz(n):    
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n, name="c")
    qc = QuantumCircuit(qr, cr)

    qc.h(0)
    for i in range(1, n):  qc.cx(0,i)

    qc.measure_all()
    return qc

#=================================
#================================= 
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
  
    nq=4
    qcP=create_ghz(nq)
    print(qcP)
    print('M: ideal circ gates count:', qcP.count_ops())
    
    qpuName='sirius'  # star 16 qubits
    qpuName='deneb'  #  diamond 20 qubits
    print('M: access IQM backend ...',qpuName)
    provider=IQMProvider(url="https://cocos.resonance.meetiqm.com/"+qpuName)
    backend = provider.get_backend()
    print('got BCKN:',backend.name,qpuName)
    
    qcT = transpile_to_IQM(qcP, backend,optimization_level=3,seed_transpiler=42)  
    cxDepth=qcT.depth(filter_function=lambda x: x.operation.name in ['cz', 'move'])
    
    print('.... PARAMETRIZED Transpiled (%s) CIRCUIT .............., cx-depth=%d'%(backend.name,cxDepth))
    print('M: transpiled gates count:', qcT.count_ops())
    print(qcT.draw('text', idle_wires=False))
    
    physQubitLayout = qcT._layout.final_index_layout(filter_ancillas=True)
    print('phys qubits:',physQubitLayout)
     
   
