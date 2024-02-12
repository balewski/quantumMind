#!/usr/bin/env python3
'''
converts qiskit to qasm  to see the syntax for strange gates
'''

import qiskit as qk
from pprint import pprint
from qiskit.circuit.library import RZXGate,RXGate,RYGate,XXPlusYYGate,CPhaseGate
import numpy as np

#...!...!....................
def ghz_circ(nq=3):
    name='ghz_%dq'%nq
    ghz = qk.QuantumCircuit(nq, nq,name=name)
    ghz.h(0)
    for idx in range(1,nq):
        ghz.cx(0,idx)
    ghz.barrier(range(nq))
    ghz.measure(range(nq), range(nq))
    return ghz

#...!...!....................
def rnd_circ(nq=3):
    name='rnd_%dq'%nq
    qc = qk.QuantumCircuit(nq, nq,name=name)
    qc.h(0)
    for idx in range(1,nq):
        qc.cx(0,idx)

    qc.append(XXPlusYYGate(np.pi),[1,0])
    qc.append(CPhaseGate(np.pi/3),[1,2])
    return qc


#=================================
#=================================
#  M A I N
#=================================
#=================================

print('\n\n ******  NEW circuit : GHZ  ****** ')
circ=rnd_circ()
print(circ)


circStr=circ.qasm()
print('M:circQasm',circStr)
#print(circStr.replace('\n','\\n'))
#print(circStr.split('\n'))

#opL=['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[2];', 'creg c[2];','xx_plus_yy(pi/2,0) q[1],q[0];','']
#opL=['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[3];', 'creg c[3];', 'gate xx_plus_yy(param0,param1) q0,q1 { rz(0) q0; rz(-pi/2) q1; sx q1; rz(pi/2) q1; s q0; cx q1,q0; ry(-pi/4) q1; ry(-pi/4) q0; cx q1,q0; sdg q0; rz(-pi/2) q1; sxdg q1; rz(pi/2) q1; rz(0) q0; }', 'h q[0];', 'cx q[0],q[1];', 'cx q[0],q[2];', 'xx_plus_yy(pi/2,0) q[1],q[0];']
#circStr= '\n'.join(opL)
circ2=qk.QuantumCircuit.from_qasm_str(circStr)
print('M:done')
print(circ2)
