import time,os,sys
from pprint import pprint
import numpy as np

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  write_yaml, read_yaml

''' Let N - number of trails,
H - number of heads
p - true  probability of heads
ps = H/N   - estimated probability fo heads

log-likelihood of measuring H heads given N trials is given by the formula:
LogL(H,N)= ln[ ps^H * (1-ps)^(N-H) ] =
     H*ln(H) - N*ln(N) + (N-H)*ln(N-H)
'''     
#...!...!....................
def LogLike(N_H):
    [N,H]=N_H
    return H*np.log(H) - N*np.log(N) + (N-H)*np.log(N-H)


''' Likelihood Ratio Test:
for a pair of coins A,B   score is defined as
  score= -2 *  ( LogL(A+B) -  LogL(A) - LogL(B) )

if H0: p_A==p_B (coins are identical) then score has chi2(dof=1) distribution
One can compute p=value of H0

'''
#...!...!....................
def Log2Log(NH_A, NH_B):
    NH_AB=[ NH_A[0]+ NH_B[0], NH_A[1]+ NH_B[1]]
    #print('hh',NH_A, NH_B)
    #print('A,B,D=',NH_A[1],  NH_B[1],  NH_A[1]- NH_B[1])
    logLa=LogLike(NH_A)
    logLb=LogLike(NH_B)
    logLab=LogLike(NH_AB)
    #print('zz', NH_AB, logLab, logLa, logLb)
    return  -2 * (logLab - logLa - logLb )

#............................
#............................
#............................
class One_Coin(object):
    def __init__(self, name,prH):
        self.name=name
        self.probHead=prH
        print('cstr:',self.name,',prob head=%.3f'%prH)
        
#...!...!....................
    def run(self,shots):
        #print(self.name,'run shots=',shots)
        x=np.random.uniform(0,1.,size=shots)
        nH=np.sum(x<self.probHead)        
        #print('num heads',nH)
        self.data=[shots,nH]
        
