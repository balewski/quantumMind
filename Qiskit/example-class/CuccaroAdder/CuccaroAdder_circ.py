__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Implements Cuccaro adder for n=1,2,...

Let a=a_2.a_1.a_0;  b=b_2.b_1.b_0
let c=a+b = c_3.c_2.c_1.c_0

Drop registers, ue just qubit IDs

Implemented based on paper:
https://arxiv.org/pdf/quant-ph/0410184.pdf
"A new quantum ripple-carry addition circuit", Steven A. Cuccaro at all

'''


#............................
#............................
def circuit_Cuccaro_n1(qc): #,qr,cr):
    #print('build n=1 circuit : MAJ+UMA adder')
    # input bit order  :  b0 a0 0 0
    # output bit order :  s0 a0 c0 c1
    # group by MAJ/UMA
    # MAJ-1
    qc.cx(1,0)
    qc.cx(1,2)
    qc.ccx(2,0,1)
    
    qc.cx(1,3)
    #return #???
    #UMA-1
    qc.ccx(2,0,1)
    qc.cx(1,2)
    qc.cx(2,0)

#............................
#............................
def circuit_Cuccaro_n2or3(qc,n):
    #print('build n=%d circuit : MAJ+UMA adder sequence'%n)
    assert n==2 or n==3
    # order : LSBF
    # input bit order  :  b0 a0 0 b1 a1 0
    # output bit order :  s0 a0 c0 s1 a1 c2
    # group by MAJ/UMA
    # MAJ-1
    qc.cx(1,0)
    qc.cx(1,2)
    qc.ccx(2,0,1)
    
    # MAJ-2
    qc.cx(4,3)
    qc.cx(4,1)
    qc.ccx(1,3,4)
    
    if n==2:
        qc.cx(4,5)
        #return #??
    else:
        # MAJ-3
        qc.cx(6,5)
        qc.cx(6,4)
        qc.ccx(4,5,6)
    
        qc.cx(6,7)

        #UMA-2
        qc.ccx(4,5,6)
        qc.cx(6,4)
        qc.cx(4,5)

    #UMA-2
    qc.ccx(1,3,4)
    qc.cx(4,1)
    qc.cx(1,3)

    #UMA-1
    qc.ccx(2,0,1)
    qc.cx(1,2)
    qc.cx(2,0)

#............................
#............................
def circuit_Cuccaro_n4plus(qc,n,A,B,X,Z):
    #print('build n=%d algorithmic circuit : MAJ+UMA sequence'%n)
    # https://arxiv.org/pdf/quant-ph/0410184.pdf  Fig 5.
    # order : LSBF
    # input bit order  :  b0 a0 0 b1 a1 0
    # output bit order :  s0 a0 c0 s1 a1 c2
        
    for i in range(1,n):
        #print('rr1',i,A[i],B[i])
        qc.cx( A[i], B[i])

    qc.cx( A[1], X)
    qc.ccx( A[0], B[0], X);   qc.cx( A[2], A[1])
    qc.ccx( X, B[1], A[1]);   qc.cx( A[3], A[2])
    
    for i in range(2,n-2):
        #print('rr2',i,A[i],B[i])
        qc.ccx( B[i], A[i-1], A[i]);  qc.cx( A[i+2], A[i+1])

    qc.ccx( A[n-3], B[n-2], A[n-2]);   qc.cx( A[n-1], Z)
    #qc.barrier()

    qc.ccx( A[n-2], B[n-1], Z)
    for i in range(1,n-1):
        #print('rr3',i,A[i],B[i])
        qc.x( B[i])

    qc.cx( X,B[1])
    for i in range(2,n):
        #print('rr4',i,A[i],B[i])
        qc.cx( A[i-1], B[i])
        
    qc.ccx( A[n-3], B[n-2], A[n-2])

    for i in range(n-3,1,-1):
        #print('rr5',i,A[i],B[i])
        qc.ccx( A[i-1], B[i], A[i])
        qc.cx( A[i+2], A[i+1])
        qc.x( B[i+1])
        
    qc.ccx( X, B[1], A[1]);  qc.cx( A[3], A[2]);  qc.x( B[2])
    qc.ccx(  B[0], A[0],X);  qc.cx( A[2], A[1]);  qc.x( B[1])

    qc.cx( A[1], X)
    for i in range(n):
        #print('rr6',i,A[i],B[i])
        qc.cx( A[i], B[i])


#...!...!....................
