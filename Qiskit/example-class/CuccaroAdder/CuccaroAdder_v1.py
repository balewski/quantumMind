__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import time,os,sys
from pprint import pprint
import numpy as np
from bitstring import BitArray

from CuccaroAdder_circ import circuit_Cuccaro_n4plus,  circuit_Cuccaro_n2or3, circuit_Cuccaro_n1


#............................
#............................
#............................
class CuccaroAdder_v1(object):

#...!...!....................
    def __init__(self,args):
        self.verb=args.verb
        self.meta={'rndSeed':args.randomSeed}
        self.meta['nbit']=args.numBits
        self.meta['nqbit']=2*args.numBits+2 # a, b,  0-line, Z-line
        self.noise=args.noise
               

#...!...!....................
    def build_circuit(self,xI,yI):        
        nbit=self.meta['nbit']
        nqbit=self.meta['nqbit']
        
        inp_x=BitArray(uint=xI, length=nbit)
        self.meta['inp_x']=inp_x
        inp_y=BitArray(uint=yI, length=nbit)
        self.meta['inp_y']=inp_y            
        
        # Step A:  Creating registers
        qc = QuantumCircuit(nqbit,nqbit)  

        # assign proper qbit lines to input B
        B=[ 1+2*i for i in range(nbit)]
        B[0]=0
        A=[ i+1 for i in B]
        X=2; Z=nqbit-1
        #print('BC:mapB=',B);  print('mapA=',A);  print('mapX=',X,' mapZ=',Z)
        self.mapABXZ=A,B,X,Z
        
        # Step B:  setup initial state  as defined by inp_x,y
        ones = inp_x.findall([1])        
        for i in ones: #MSBF , i=True if bit is 1
            iq=A[nbit-i-1]
            #print('xo',i,nbit-i-1,iq)
            qc.x(iq) #LSBF

        ones = inp_y.findall([1])        
        for i in ones: #MSBF
            iq=B[nbit-i-1]
            #print('yo',i,nbit-i-1,iq)
            qc.x(iq) #LSBF

        if self.noise>0: # should be used only for simulator
            for i in range(nqbit):
                qc.ry(self.noise*np.pi,i)

        # Step C:  define circuit for Va
        qc.barrier() 
        if nbit==1:
            circuit_Cuccaro_n1(qc)
        elif nbit==2 or nbit==3:
            circuit_Cuccaro_n2or3(qc,nbit) 
        else:
            circuit_Cuccaro_n4plus(qc,nbit,A,B,X,Z)  
        
        # Step D: perform measurement on the all registers
        qc.measure_all(add_bits=False)
        # ???If add_bits=False, the results of the measurements will instead be stored in the already existing classical bits

        tot_gates=qc.size()
        if self.verb>1 : print('oracle tot_gates=',tot_gates)
        self.meta['nGate']=tot_gates
        
        self.meta['qasmSoft']=qc.qasm().split('\n')        
        self.qc=qc

 


#===========================================
#  FUNCTIONS 
#===========================================
#...!...!....................
def eval_adder_v1(jobD,circD,infoD,par_nSig=3):
    print("classic eval of CuccaroAdder_v1")
    A,B,X,Z=infoD['mapABXZ']
    pprint(infoD)
    nbit=infoD['info']['nbit']
    nqbit=infoD['info']['nqbit']

    print('\nEval Cuccaro for all inputs:')
    nOK=0
    for iR in jobD:
        job=jobD[iR]
        result = job.result()
        md=circD[iR].meta
        #pprint(md)
        countsD = result.get_counts(0);  #print('cD:',countsD)
        assert len(countsD)==1
        
        inp_x=md['inp_x']
        inp_y=md['inp_y']
        #print('kk',inp_x,inp_y,type(inp_x))
        xI=inp_x.uint
        yI=inp_y.uint
        zTrue=xI+yI  # this is correct answer
        #print('truth: xI,yI,zTrue=',xI,yI,zTrue)
       
        S=list(countsD)[0]
        
        #print( 'S(MSBF): ', S, type(B[0]),type(S))

        fbin3=[S[nqbit-x-1] for x in  B]
        fbin3.append(S[0]) # add highest carry bir
        fbin3=''.join(reversed(fbin3)) # inverse order to get MSBF
        #print('fbin3',fbin3,', mapB:',B)
        fx=BitArray(bin=fbin3).uint
        if zTrue==fx: nOK+=1
        if iR<15:
            print("iR=%d  f(%s,%s) = %s %d=T R=%s, %s"%(iR,inp_x,inp_y,fx,zTrue,S,fx==zTrue))
        else :
            print('.',end='')

    
    if nOK==len(jobD):
        print('* * * SUCCESS * * *',nOK)
    else:
        print("# # # WRONG # # #")

   
  
