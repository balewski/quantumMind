__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' module correcting for readout error of speciffic MIS state
'''
from bitstring import BitArray
import numpy as np
from pprint import pprint

#............................
#............................
#............................

class ReadErrorMitigation4MIS():
#...!...!....................
    def __init__(self, task):
        self.verb=task.verb
        self.expD=task.expD
        

#...!...!....................
    def find_similar(self,trgstr):  # use MSBL convention
        A=BitArray(bin=trgstr)
        misID=A.uint
        wA=A.count(True) # Hamming weight
        print('trg',trgstr,misID,wA)

        #... generate all bistrings differing by one
        nBit=len(trgstr)

        upL=[]; downL=[]
        for i in range(nBit):
            B=A.copy()
            B[i]=not B[i]
            wB=B.count(True)
            wDel=wA-wB
            print('i=',i,A,B,A^B,wB,wDel)
            if wDel==1: downL.append(B.uint)
            if wDel==-1: upL.append(B.uint)
        print('found   upL:',upL)
        print('found downL:',downL)
        self.upL=upL
        self.downL=downL
        self.target=A.uint

#...!...!....................
    def find_measurements(self):
        misIDs=self.expD['ranked_MISs']  # [NX,IdCa]
        #self.expD['ranked_counts']=dataY # [NX,NB]
        dataP=self.expD['ranked_probs'] # [PEY,NX,NB]
        #
        pprint(misIDs[:,0])
        idx=np.where(misIDs[:,0] == self.target)[0][0]
        acc=dataP[:,idx,1].copy()
        print('target raw:',idx,acc)
        acc[1]=acc[1]**2 # now it is variance
        for mid in self.downL:
            idxL=np.where(misIDs[:,0] == mid)[0]
            if len(idxL)==0: continue
            idx=idxL[0]
            pev=dataP[:,idx,1]
            print('found idx:',idx,'PEV:',pev)
            acc[0]+=pev[0]  # P
            acc[1]+=pev[1]**2  # V
            acc[2]+=pev[2]  #Y
        acc[1]=np.sqrt(acc[1])
        print('target corrected:',acc)
