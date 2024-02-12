#!/usr/bin/env python3
'''
based on https://trueq.quantumbenchmark.com/examples/protocols/srb.html#changing-the-twirling-group-gateset

Exmaplation
Streamlined Randomized Benchmarking (SRB)

1 qubit
use sequence lengths [4, 30, 50], and, by default, 30 random circuits 

Goal:  verify Uinv is indeed the invers of product of all U3 in the circuit
'''
import trueq as tq
import numpy as np
from pprint import pprint
import trueq.simulation as tqs
print('True-Q ver',tq.__version__)
np.set_printoptions(precision=2)

if 0:
    pp=tq.make_srb
    print('Inspect',pp.__name__,type(pp),pp.__code__.co_argcount,'args:' ,pp.__code__.co_varnames)

qid=5 # qubit id must match gates-.yaml
circCol = tq.make_srb(qid, [2,10,20],n_circuits=5, twirl="U")
circ=circCol[0]
print('M:circ',type(circ),len(circ),'circCol len=',len(circCol))

m=[]
for i,cycle in enumerate(circ):
    gates=cycle.gates
    if len(gates)<1: continue
    print('M:cycle=',i,cycle,list(gates.keys()))
    for qidg in gates.keys(): # 1 or 2 qubit tuple
        gate=cycle.gates[qidg]
        print('  qidg',qidg,gate.mat)
        m.append(gate.mat)

# check if product is identity
x=m[0]
for y in m[1:]:
    x=np.matmul(y,x)
print('\nM:prod1');pprint(x)

#gateF='trueq_gate_X6Y8.yaml'
gateF='trueq_gate_chip57.yaml'

print('\nM: transpile to QubiC basis gates and use chip mapping:',gateF)
config=tq.Config.from_yaml(gateF)
print('trueq_gatecfg',config)


transpiler = tq.Compiler.from_config(config)
transCol = transpiler.compile(circCol)
print('M: transpile done')

circ=transCol[0]
print('M:circ',type(circ),len(circ),'circCol len=',len(transCol))
m=[]
for i,cycle in enumerate(circ):
    gates=cycle.gates
    if len(gates)<1: continue
    #print('M:cycle=',i,cycle,list(gates.keys()))
    for qidg in gates.keys(): # 1 or 2 qubit tuple
        gate=cycle.gates[qidg]
        #print('  qidg',qidg,'matrix=',gate.mat)
        print('  qidg',qidg,'name=',gate.name.upper(),gate.parameters) 
        m.append(gate.mat)
# check if product is identity
x=m[0]
for y in m[1:]:
    x=np.matmul(y,x)
print('\nM:prod2');pprint(x)




 

