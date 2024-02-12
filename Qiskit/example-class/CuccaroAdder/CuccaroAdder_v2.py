__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
note: _v2   uses september convention for  qubits-clbits mapping: 
qubits 0,1,...,N-1 will map to the bits of the integer representation : b_0, b_1, …, b_(N-1)  , where 0 is LSB, and N-1 is MSB
allows for Binary Representation Equivalent for Toffoli --> CX

 for Cuccaro adder there are 2 operands: A,B and address.
 Operand A uses  lower ids qubits, operand B uses higher ids qubits. 
the 2 ancilla qubits are the MSB bits of output A,B
Diagram for sum:  (0,A,0,B) → (0,A, B+A); for diff : (0,A,B*) → (A, B*-A), where '*' denotes additional MSB set on input 


'''

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
import time,os,sys
from pprint import pprint
import numpy as np
from bitstring import BitArray

from CuccaroAdder_circ import circuit_Cuccaro_n4plus,  circuit_Cuccaro_n2or3, circuit_Cuccaro_n1


from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout

#Convention:  the variables carying bit-wise data are stored as BitArray class, see https://pythonhosted.org/bitstring/



#...!...!....................
def remap_qubits(qc,targetMap):
    # access quantum register
    qr=qc.qregs[0]
    nq=len(qr)
    assert len(targetMap)==nq
    #print('registers has %d qubist'%nq,qr)
    regMap={}
    for i,j in enumerate(targetMap):
        #print(i,'do',j)
        regMap[qr[j]]=i

    #print('remap qubits:'); print(regMap)
    layout = Layout(regMap)
    #print('layout:'); print(layout)

    # Create the PassManager that will allow remapping of certain qubits
    pass_manager = PassManager()

    # Create the passes that will remap your circuit to the layout specified above
    set_layout = SetLayout(layout)
    apply_layout = ApplyLayout()

    # Add passes to the PassManager. (order matters, set_layout should be appended first)
    pass_manager.append(set_layout)
    pass_manager.append(apply_layout)

    # Execute the passes on your circuit
    remapped_circ = pass_manager.run(qc)

    return remapped_circ


#............................
#............................
#............................
# define my own class overwriteing cxx gate which works for binary encodings
#  Binary Representation Equivalent for Toffoli --> CX
class QuantumCircuit_BRE(QuantumCircuit):
  def __init__(self,nqbit,ncbit=None):
    QuantumCircuit.__init__(self,nqbit,ncbit)
  def ccx(self,i,j,k):
    self.x(i)
    self.x(j)
    
    for m in range(2):
      self.ry(np.pi/4,k)
      self.cx(j,k)
    
      self.ry(np.pi/4,k)
      self.cx(i,k)
      
    self.x(i)
    self.x(j)
 

#............................
#............................
#............................
class CuccaroAdder_v2(object):

#...!...!....................
    def __init__(self,args):
        self.verb=args.verb
        self.meta={}
        self.meta['num_operand_bit']=args.numBits
        self.meta['num_qubit']=2*args.numBits+2 # a, b,  0-line, Z-line
        self.meta['math']='adder'
        self.meta['bre_transf']=args.doBRE # transforms Toffoli to 4 CX
        self._build_adder_circuit()
        if 1:  # remap order of qubits
            A,B,X,Z=self.mapABXZ
            trgMap=A+[X]+B+[Z]  # new order of qubits , B+[Z] will be the output sum
            print('remap: trgMap',trgMap)
            qc=remap_qubits(self.circ,trgMap)
            print('Final adder qubit map, MSBF input: [0,B,0,A], output:[A+B,0,A]')
            # Uses consecutive qubits for operands  
            # MSBF data bit assigned to highest qubit
            self.circ=qc

        if args.doGradient:
            print('change adder to gradient')
            self.meta['math']='grad'
            self.circ=self.circ.inverse()

            
#...!...!....................
    def _build_adder_circuit(self):
        nbit=self.meta['num_operand_bit']
        nqbit=self.meta['num_qubit']
        
        # Step A:  Creating registers
        if self.meta['bre_transf']: # transforms Toffoli to 4 CX
            qc = QuantumCircuit_BRE(nqbit,nqbit)
        else:
            qc = QuantumCircuit(nqbit,nqbit)
        # assign proper qbit lines to input B
        B=[ 1+2*i for i in range(nbit)]
        B[0]=0
        A=[ i+1 for i in B]
        X=2; Z=nqbit-1
        print('BC:mapB=',B);  print('mapA=',A);  print('mapX=',X,' mapZ=',Z)
        self.mapABXZ=A,B,X,Z
        
        # Step C:  define circuit for adder
        
        if nbit==1:
            circuit_Cuccaro_n1(qc)
        elif nbit==2 or nbit==3:
            circuit_Cuccaro_n2or3(qc,nbit)
        else:
            circuit_Cuccaro_n4plus(qc,nbit,A,B,X,Z)
        
        self.circ=qc

#...!...!....................
    def iniState(self,xI,yI):
        nbit=self.meta['num_operand_bit']
        nqbit=self.meta['num_qubit']
        nqh=nqbit//2
        
        Bx=BitArray(uint=xI,length=nbit)
        By=BitArray(uint=yI,length=nbit)
        
        qc = QuantumCircuit(nqbit,nqbit)

        # ... store A
        Bx.reverse()
        for i in range(nbit):
            if Bx[i]: qc.x(i)
            
        # ... store B
        By.reverse()
        for i in range(nbit):
            j=nqh+i
            if By[i]: qc.x(j)
        qc.barrier()
        return qc
            


#===========================================
#  FUNCTIONS 
#===========================================

#...!...!....................
def eval_adder_v2(jobD,md,inpD,par_nSig=3):
    print("classic eval CuccaroAdder_v2 as adder")
    #    pprint(md)
    nbit=md['num_operand_bit']
    nqbit=md['num_qubit']
    nqh=nqbit//2

    njob=len(jobD)
    print('\nEval Cuccaro %s for all %d inputs:'%(md['math'],njob))
    nOK=0
    for iR in jobD:
        job=jobD[iR]
        result = job.result()
        countsD = result.get_counts(0);  #print('cD:',countsD)
        assert len(countsD)==1
        
        xI,yI=inpD[iR]
        assert len(countsD)==1 ,'only one bitstring is expected '
        M=list(countsD)[0]  # the only measurement
        #print('meas=',M,M[:nqh],M[-nqh:])
        dB=BitArray(bin=M[-nqh:]) # this value is the  same as inpu xI 
        zB=BitArray(bin=M[:nqh])  # this is the result (sum or difference)
        
        if md['math']=='adder':            
            zTrue=xI+yI  # this is correct answer
            #print(iR,'truth: xI,yI,zTrue=',xI,yI,zTrue)
            zI=zB.uint
            trueStr='%d+%d=%d'%(xI,yI,zI)
        
        if md['math']=='grad':            
            zTrue=yI-xI  # this is correct answer
            #print(iR,'truth: xI,yI,zTrue=',xI,yI,zTrue)
            zI=zB.int
            trueStr='%d-%d=%2d'%(yI,xI,zTrue)
        
        
        if zTrue==zI: nOK+=1
        if iR<16:
            print("iR=%2d   %s  %2d=T   M=%s, %s, d=%d"%(iR,trueStr,zI,M,zI==zTrue,dB.uint))
        else :
            print('.',end='')
            if iR%50==0 : print(iR,end='')
        
    if nOK==len(jobD):
        print('* * * SUCCESS * * *',nOK)
    else:
        print("# # # WRONG # # #")

   
