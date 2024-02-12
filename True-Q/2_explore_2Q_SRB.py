#!/usr/bin/env python3
'''
based on https://trueq.quantumbenchmark.com/examples/protocols/srb.html#changing-the-twirling-group-gateset

Exmaplation
Streamlined Randomized Benchmarking (SRB)

2 qubit

'''
import trueq as tq
import numpy as np
from pprint import pprint
import trueq.simulation as tqs
print('True-Q ver',tq.__version__)
np.set_printoptions(precision=2)

# do 2 single-qubit SRB
circCol = tq.make_srb([[3],[5]], [1,10,20],n_circuits=5, twirling_group="SU")

circ=circCol[0]
print('M:circ',type(circ),len(circ),'circCol len=',len(circCol))

m={}
for i,cycle in enumerate(circ):
    gates=cycle.gates
    if len(gates)<1: continue    
    print('M:cycle=',i,cycle,list(gates.keys()))
    for qidg in gates.keys(): # 1 or 2 qubit tuple
        if qidg not in m: m[qidg]=[]
        gate=cycle.gates[qidg]
        print('  qidg',qidg);pprint(gate.mat)
        assert len(qidg)==1 # matmul encoded only for disjoint qubits
        m[qidg].append(gate.mat)
# check if product is identity
for qidg in m:
    x=m[qidg][0]
    for y in m[qidg][1:]:
        x=np.matmul(y,x)
    print('\nM:prod1',qidg);pprint(x)


gateF='trueq_gate_X6Y8.conf'
print('\nM: transpile to QubiC basis gates and use chip mapping:',gateF)
f=open(gateF)
trueq_gatecfg=f.read()
f.close()
#1print('M:trueq_gatecfg', config)
config=tq.Config(trueq_gatecfg)

transpiler = tq.compilation.get_transpiler(config)
transCol = transpiler.compile(circCol)
print('M: transpile done')

circ=transCol[0]
print('M:circ',type(circ),len(circ),'circCol len=',len(transCol))
for qidg in m: m[qidg]=[] # clear but keep qibit ids

for i,cycle in enumerate(circ):
    gates=cycle.gates
    if len(gates)<1: continue
    #print('M:cycle=',i,cycle,list(gates.keys()))
    for qidg in gates.keys(): # 1 or 2 qubit tuple
        gate=cycle.gates[qidg]
        #print('  qidg',qidg,'matrix=',gate.mat)
        print('  qidg',qidg,'name=',gate.name.upper(),gate.parameters) 
        m[qidg].append(gate.mat)
# check if product is identity
for qidg in m:
    x=m[qidg][0]
    for y in m[qidg][1:]:
        x=np.matmul(y,x)
    print('\nM:prod2',qidg);pprint(x)





 

