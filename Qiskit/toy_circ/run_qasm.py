#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Execute Qasm-defined circuit on local simulator

required INPUT:
--circName: ghz_5qm

'''
import time,os,sys
from qiskit import QuantumCircuit, Aer,  execute, transpile, assemble
from pprint import pprint

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='circuits',help="input for everything")

    parser.add_argument("-c","--circName", default='grover_5Qs011i2',
                        help="name of ideal  circuit")
    parser.add_argument('-O','--optimLevel',type=int,default=1, help="transpiler optimization 0:just converts gates, 1:agregates , 2: changes mapping")
    parser.add_argument('-r',"--rnd_seed",default=123,type=int,help="for transpiler")
    parser.add_argument('-s','--shots',type=int,default=8192, help="shots")


    args = parser.parse_args()
    args.dataPath+='/' 

    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    for xx in [args.dataPath]:
        assert  os.path.exists(xx)
    return args


#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

circF=args.dataPath+'/'+args.circName+'.qasm'
circOrg=QuantumCircuit.from_qasm_file( circF )
print('\ncirc original: %s, depth=%d'%( circF,circOrg.depth()))
if args.verb>0:
    print(circOrg)

print("\n ------- Use Aer's qasm_simulator for counting")
backend = Aer.get_backend('qasm_simulator')
assert backend.configuration().simulator

print(' Layout using optimization_level=',args.optimLevel,backend)

print("\n ------- Transpile circuit to base-gates")
circOpt = transpile(circOrg, backend=backend,optimization_level=args.optimLevel, seed_transpiler=args.rnd_seed, basis_gates=['u3','cx'])
print('circ transpiled, seed=%d depth=%d'%(args.rnd_seed, circOpt.depth()))
if args.verb>1:
    print(circOpt)

qobj_circ = assemble(circOpt, backend, shots=args.shots)
if args.verb>2:
    print('\nqobj_circ'); pprint(qobj_circ.to_dict())

print('executes, shots=%d'%args.shots)
job =  backend.run(qobj_circ)

# Grab the results from the job.
result = job.result()
counts = result.get_counts(0)
print('counts for %s:'%args.circName)
pprint(counts)
