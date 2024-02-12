__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

''' module correcting for readout error 
assume only 1-state can decay to 0-state

PROBLEMS not solved:
* main:  what fraction of child counts should be goven to the parent?
   e.g.  there many be many  solutions with the same large HW and they may differ by HD=2  .  How to find the down-feed fraction?
A: compute median of shots, compute everage of the left half, subtract it from the right half .
This will also clear the double-counting issue

* two cliques may have overlaping child -> counts from this child should be split evenly
* used counts for a child should be removed form this child to preserve tot number of counts


TODO - confirem correlation width is not affected by readerrEps=0.08

'''
from bitstring import BitArray
import numpy as np
from pprint import pprint
from collections import Counter
from toolbox.Util_stats import do_yield_stats


#............................
#............................
#............................

class MitigateReadErrorQuEra():
#...!...!....................
    def __init__(self, task):
        self.verb=task.verb
        self.expD=task.expD
        self.meta=task.meta
        
        self._insertReadError()
        return
        self._split_by_Hamming_weight()
        
        print('MRE rannked by HW mshot:')
        for hw in sorted(self.ranked_by_HW):
            solL=self.ranked_by_HW[hw]
            print('hw=%d  nSol=%d, solL'%(hw,len(solL)),solL)
        
#...!...!....................
    def _insertReadError(self,readErrEps=0.08):
        amd=self.meta['analyzis']
        amd['add_readErr_eps']=readErrEps
        nAtom=amd['num_atom']
        nSol=amd.pop('num_sol')
        #... deattach current data 
        dataY=self.expD.pop('ranked_counts') # [NB]
        hexpattV=self.expD.pop('ranked_hexpatt')
        hammwV=self.expD.pop('ranked_hammw')
        self.expD.pop('ranked_probs')
        
        cntD={}
        #print('eedd',sorted(self.expD))
        
        def spread_measurements(hexpatt,mshot):
            A=BitArray(hex=hexpatt) # keep leading 0s to retain hex-capability
            oneIdxL=[  i  for i, bit in enumerate(A) if  bit]
            #print('parent=',hexpatt,A.bin, 'mshot=',mshot, 'oneIdxL:',oneIdxL)

            #mshot=100   
            for j in range(mshot):
                C=A.copy()
                #np.random.shuffle(oneIdxL)  # maybe over-kill?
                for i in oneIdxL:  # conditional bit-1 flip
                    if np.random.uniform()< readErrEps: C[i]=0                    
                patt=C.hex
                #print(j,A,'C=',C.bin,'k=',k)
                if patt not in cntD: cntD[patt]=0
                cntD[patt]+=1  # add one shot

        accPattL=['15555'] # rgrgr... card=9
        for hexpatt,mshot in zip( hexpattV,dataY):
            #if hexpatt not in accPattL: continue
            spread_measurements(hexpatt,mshot)
                        
            
        print('cntD nSol=%d'%(len(cntD))); #print(cntD)
        
        self._update_expD(cntD)
        
#...!...!....................
    def _update_expD(self,cntD):  # similar to end of raw_experiment_analysis(.)
        amd=self.meta['analyzis']
        nAtom=amd['num_atom']
        uShots=amd['used_shots']
        nSol=len(cntD)
        solCounter=Counter(cntD)
    
        #...... update meta-data
        amd['num_sol']=nSol
        rankedSol=solCounter.most_common(nSol)
        
        NB=nSol # number of bitstrings,  this is sparse encoding 
        dataY=np.zeros(NB,dtype=np.int32)  # measured shots per pattern
        pattV=np.empty((NB), dtype='object')  # meas bitstrings as hex-strings (hexpatt)
        hammV=np.zeros(NB,dtype=np.int32)  # hamming weight of a pattern

        print('dataY:',dataY.shape)
        for i in range(nSol):
            hexpatt,mshot=rankedSol[i]
            A=BitArray(hex=hexpatt) # keep leading 0s
            hw=A.bin.count('1')  # Hamming weight
            if i<10: print('u:ii',i,'hex=%s hw=%d mshot=%d'%(A.hex,hw,mshot))
            dataY[i]=mshot
            pattV[i]=A.hex
            hammV[i]=hw
            
        if self.verb>1:
            print('u:dataY:',dataY)
            print('u:pattV',pattV)
            print('u:hammV',hammV)

        self.expD['ranked_counts']=dataY   # [NB]
        self.expD['ranked_hexpatt']=pattV  # [NB]
        self.expD['ranked_hammw']=hammV    # [NB]

        dataP=do_yield_stats(dataY.reshape(1,-1)).reshape(3,-1)
        if self.verb>1:
            print('u:dataP:',dataP)
        self.expD['ranked_probs']=dataP   # [PEY,NB]

        
#...!...!....................
    def _split_by_Hamming_weight(self):
        outD={}
        for [solId,hw],mshot in zip(self.expD['ranked_IdHw'], self.expD['ranked_counts'][:,1]):
            #print('solId:',solId,hw,mshot)
            if hw not in outD: outD[hw]={}
            outD[hw][solId]=mshot
        #self.expD['ranked_by_HW_meas']=
        self.ranked_by_HW=outD


        

#...!...!....................
    def mitigate(self):
        amd=self.meta['analyzis']
        nAtom=amd['num_atom']
        minNumChild=3
        inpDD=self.ranked_by_HW
        hwL=sorted(inpDD)
        print('mitS:hwL',hwL)
        def find_1clique(hw,inpD): # groupe by Hamming dist =2
            print('find_clique(hw=%d, inpD.len=%d'%(hw,len(inpD)))
            A, cnts = inpD.pop()
            print('1st clique hw=%d mID=%d %s cnts=%d'%(hw,A.uint,A.bin,cnts))
            cliqL=[ [A,cnts]]
            leftL=[]
            nI=len(inpD)
            while len(inpD)>0:
                B,cnts = inpD.pop()
                X=A^B
                hw=X.count('1')  # Hamming weight
                #print('  A=%s vs. B=%s X=%s w=%d'%(A.bin,B.bin,X.bin,hw))
                if hw==2: cliqL.append([B,cnts])
                else: leftL.append([B,cnts])
            print('found clique size=%d :'%len(cliqL),cliqL)
            return cliqL,leftL

        cliqLL=[]
        for hw in hwL:  # hw needs to increase to correct counts for the next layer
            if hw<4: continue  # do not deal w/ low HW strings
            #... convert all IDs to BitArr for effiviency once
            inpA=[ [BitArray(uint=int(mID),length=nAtom), cnts] for mID, cnts in inpDD[hw].items() ]            
            print('hw-loop hw=%d, inpA:'%(hw),inpA)
            while len(inpA)>0:
                clique,inpA=find_1clique(hw,inpA)
                if len(clique)>=minNumChild: cliqLL.append(clique)
            print('found total num cliques=%d'%(len(cliqLL)))

            inpE=inpDD[hw+1]
            for clique in cliqLL:
                parId,sumShot=self._mitigate_clique(clique,minNumChild)
                if sumShot<=0: continue
                print('mit hw=%d add shots=%d'%(hw,sumShot),'\n')
                
                if parId not in inpE: inpE[parId]=0
                inpE[parId]+=sumShot
                #break
            print('updated hw=%d layer:'%(hw+1),inpE)
        aa4
#...!...!....................
    def _mitigate_clique(self,clique,minNumChild):
        amd=self.meta['analyzis']
        nAtom=amd['num_atom']
        
        tmpD={}
        
        while len(clique)>0:
            A,mshot = clique.pop()
            print('child=',A.uint,A.bin, 'mshot=',mshot)
            # try to replace every 0 -->1, one at a time
            for i, bit in enumerate(A):                
                if bit : continue
                C=A.copy()
                C[i]=1
                k=C.uint
                #print('mit:',i,A,bit,'C=',C.bin,'k=',k)

                if k not in tmpD: tmpD[k]=[]
                tmpD[k].append((A.uint,mshot))
        #break
        print('mit  tmpD',tmpD)

        # ... accept only prarent with several children
        
        idL=sorted(tmpD)
        for solId in idL:
            if len( tmpD[solId] ) < minNumChild: tmpD.pop(solId)
        if len(tmpD)<=0: return -1,0
        parId,resL=tmpD.popitem()    
        print('accepted parent=%d childL:'%parId,resL)
        totMshot=0
        for mID,mshot in resL:
            totMshot+=mshot
        print('totMshot=',totMshot)
        return parId,totMshot

        
