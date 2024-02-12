#!/usr/bin/env python3
'''
converts quil circuit to casm
based on DAG
only limitted set of gates is encoded
'''

import qiskit as qk
from qiskit.converters import circuit_to_dag, dag_to_circuit
from pprint import pprint
from pyquil import Program, get_qc
from pyquil.quilbase import Declare
from pyquil.gates import CNOT, Z, MEASURE,RY,H
from functools import reduce
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit

#...!...!....................
def layers_2_dict(qdag):
    print('\n Order gates chronologically  (per layer) as list of lists')
    outLL=[]
    dagLay=qdag.layers()
    n2qCyc=0
    couplerS=set()
    qubitS=set()
    measQ=[]
    for k,lay in enumerate(dagLay):      
        gateL=[]
        mxq=-1
        for op in lay['graph'].op_nodes():
            qL=[qr._index  for qr in  op.qargs]  # qubits as str
            cL=[cr._index  for cr in  op.cargs]  # clbits  as int
            pL=['%6f'% x for x in op.op._params]
            nq=len(qL)
            npa=len(pL)
            if nq>1 : couplerS.add(tuple(sorted(qL)))
            [qubitS.add(i) for i in qL]
            if mxq<nq: mxq=nq  # find longest gate in a cycle
            if 'xx_plus_yy_pi' in op.name: op.name='xx_plus_yy_pi'
            if 'measure' ==op.name: measQ.append(qL[0])
            one={'name':op.name}
            one['qubit']= qL
            #one['param']=op.op._params
            gateL.append(one)
        if mxq<2 :
            outLL.append('%d 1-q gate(s)'%len(gateL))
            continue  # only keep 2-gate cycles
        outLL.append(gateL)
        n2qCyc+=1
    pprint(outLL)
    print('2-qubit gats cycles:',n2qCyc)
    print('num couplers=%d, list:'%len(couplerS),sorted(couplerS))
    print('used  qubits=%d, list:'%len(qubitS),sorted(qubitS))
    print('meas  qubits=%d, list:'%len(measQ),sorted(measQ))
    

#...!...!....................
def shotM_2_counts(shot_meas):
    results = list(map(lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""), shot_meas))
    counts = dict(zip(results,[results.count(i) for i in results]))
    return counts


#...!...!....................
def get_qprog1(shots = 5):
    
    p=Program(
        Declare("ro", "BIT", 2),
        Z(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    )
    p.wrap_in_numshots_loop(shots)
    return p
        
#...!...!....................
def get_qprog2(shots = 1024):
    
    p = Program('PRAGMA INITIAL_REWIRING "PARTIAL"')  # Maximize program execution fidelity
    #p = Program('PRAGMA INITIAL_REWIRING "GREEDY"')  # Faster qubit allocation at expense of fidelity
    ro = p.declare('ro', memory_type='BIT', memory_size=6)

    p.inst(RY(np.pi / 4, 0))
    p.inst(RY(np.pi / 4, 1))
    p.inst(RY(np.pi / 4, 2))
    p.inst(RY(np.pi / 4, 3))
    p.inst(H(4))
    p.inst(H(5))
    p.inst(CNOT(4, 3))
    p.inst(CNOT(5, 2))
    p.inst(CNOT(4, 1))
    p.inst(CNOT(5, 0))
    p.inst(RY(np.pi / 4, 0))
    p.inst(RY(-np.pi / 4, 1))
    p.inst(RY(-np.pi / 4, 2))
    p.inst(RY(-np.pi / 4, 3))
    p.inst(CNOT(5, 3))
    p.inst(CNOT(4, 2))
    p.inst(CNOT(5, 1))
    p.inst(CNOT(4, 0))
    p.inst(RY(np.pi / 4, 0))
    p.inst(RY(-np.pi / 4, 1))
    p.inst(RY(-np.pi / 4, 2))
    p.inst(RY(np.pi / 4, 3))
    p.inst(CNOT(4, 3))
    p.inst(CNOT(5, 2))
    p.inst(CNOT(4, 1))
    p.inst(CNOT(5, 0))
    p.inst(RY(np.pi / 4, 0))
    p.inst(RY(np.pi / 4, 1))
    p.inst(RY(np.pi / 4, 2))
    p.inst(RY(-np.pi / 4, 3))
    p.inst(CNOT(5, 3))
    p.inst(CNOT(4, 2))
    p.inst(CNOT(5, 1))
    p.inst(CNOT(4, 0))
    p.inst(MEASURE(0, ro[0]))
    p.inst(MEASURE(1, ro[1]))
    p.inst(MEASURE(2, ro[2]))
    p.inst(MEASURE(3, ro[3]))
    p.inst(MEASURE(4, ro[4]))
    p.inst(MEASURE(5, ro[5]))

    p.wrap_in_numshots_loop(shots)
    return p
    
  
def quil_2_qasm(qprg):
    quilProgStr=str(qprg)
    print('\n parse quil program:')
    print(quilProgStr)
    #print(quilProgStr.replace('\n','\\n'))
    recL=quilProgStr.split('\n')

    outL=['OPENQASM 2.0;','include "qelib1.inc";','QREG-tmp','CREG-tmp']
    # code taken from https://qiskit.org/documentation/_modules/qiskit/circuit/library/standard_gates/xx_plus_yy.html#XXPlusYYGate
    outL.append('gate xx_plus_yy(theta) a, b {  rz(-pi/2) a;  sx a;  rz(pi/2) a; s b;  cx a, b; ry(theta/2) a;   ry(theta/2) b;  cx a, b;   sdg b;  rz(-pi/2) a;   sxdg a; rz(pi/2) a; }')
    outL.append('gate xx_plus_yy_pi a, b {  rz(-pi/2) a;  sx a;  rz(pi/2) a; s b;  cx a, b; ry(pi/2) a;   ry(pi/2) b;  cx a, b;   sdg b;  rz(-pi/2) a;   sxdg a; rz(pi/2) a; }')  # special case, but it does not shortens the QASM output
    mxQbit=-1
    for rec in recL:
        if len(rec)==0: continue
        if 'PRAGMA' in rec: continue
        if 'DECLARE ro BIT' in rec:
            xx=rec.split('[')[1][:-1]
            print('xx',xx)
            outL[3]='creg c[%d];'%int(xx)
            continue
        xL=rec.split(' ')
        
        name=xL[0].lower()
        if name=='halt' : break
        qL=['q[%s]'%x for x in xL[1:]]
        print('name1=',name,qL)
        if 'measure' ==name:
            assert len(xL)==3
            print('m xL',xL)
            qid=xL[1]
            cid=xL[2].split('[')[1][:-1]
            txt='measure q[%s] -> c[%s];'%(qid,cid)
        else:
            if name=='cnot': name='cx'
            if 'cphase' in name : name=name.replace('cphase','cp')
            
            if 'xy(pi)'==name: name='xx_plus_yy_pi'
            if 'xy'==name[:2]: name=name.replace('xy(','xx_plus_yy(')  # orders matters
   
            #...  parse qubit ids
            txt=name+' '+','.join(qL)+';'
            iQL=[int(x) for x in xL[1:]]  
            iqm=max(iQL) # track max qubit Id
            if mxQbit<iqm : mxQbit=iqm

        print('txt=',txt)
        outL.append(txt)
        
    outL[2]='qreg q[%d];'%(mxQbit+1)
    print('END:',outL)
    outStr='\n'.join(outL)
    return outStr
  

#=================================
#=================================
#  M A I N
#=================================
#=================================

qprg=get_qprog2()

if 0:
    print('M: run pyQuil simple')
    device_name = '6q-qvm'
    qback = get_qc(device_name, as_qvm=True)  #backend
    shot_meas = qback.run(qprg).readout_data.get("ro")
    #print('M:shot_meas',shot_meas)
    counts= shotM_2_counts(shot_meas)
    print(counts)
    exit(0)
    
if 1:
    print('M: apply Quil transpiler')
    device_name = 'Aspen-11'
    #device_name = 'Aspen-M-2'
    qback = get_qc(device_name, as_qvm=True)
    qprgT=qback.compile(qprg)
    qprg=qprgT
    

if 0:
    print('M: run pyQuil as on Aspen-11')
    shot_meas = qback.run(qprgT).readout_data.get("ro")
    #print('M:shot_meas',shot_meas)
    counts= shotM_2_counts(shot_meas)
    print(counts)
    exit(0)
    

qasmStr=quil_2_qasm(qprg)
circ=qk.QuantumCircuit.from_qasm_str(qasmStr)

outF='out123.qasm'
circ.qasm(filename=outF)
print('M: saved',outF)

print(circ)
qdag = circuit_to_dag(circ)
layers_2_dict(qdag)

exit(0)  # to skip qiskit simu
shots=200
print('M: run Qiskit simu ...',shots)

backend = qk.Aer.get_backend('qasm_simulator')
job = qk.execute(circ, backend, shots=shots)
result= job.result()
counts = result.get_counts(0)
print('counts=',counts)
print('M: saved2',outF)
print('M:done')
