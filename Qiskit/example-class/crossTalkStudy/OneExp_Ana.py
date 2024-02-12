import time,os,sys
from pprint import pprint
import numpy as np

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  write_yaml, read_yaml, npAr_avr_and_err

#...!...!....................
def import_yieldArray( jid6, dataPath):
        inpF=dataPath+'yieldArray-%s.yaml'%jid6
        blob=read_yaml(inpF)
        print('blob keys:', blob.keys())
        yields=blob.pop('yields')
        metaD=blob
        print('metaD:', metaD.keys())
        print('see %d experiments'%len(yields))
        return metaD,yields

#............................
#............................
#............................
class OneExp_Ana(object):
    def __init__(self, expId,metaD):
        self.expId=expId
        xL=expId.split('_') # must match content of experiment name
        self.prepState=int(xL[3][2:])
        self.delayTicks=int(xL[2][2:])
        self.targetQ=int(xL[1][1:])
        self.measLab=np.array(metaD['retrInfo']['measLabInt'])
        self.jid6L=[]  # keep track of all jobs
        self.execTime=[]
        self.Y=[]
        self.shots=metaD['expInfo']['shots']
        self.metaD=metaD
        print('cstr:',self.expId, ',prep:',self.prepState,',measLab:',self.measLab,',Q:',self.targetQ,',nT:',self.delayTicks)
        
#...!...!....................
    def unpack_yieldArray(self,jid6, execDate, yields):
        self.jid6L.append(jid6)
        self.execTime.append(execDate)
        #print('ny',len(yields),self.expId )
        # note, same expId is present mutiple times
        nCyc=0
        for expD in yields:
            
            if expD['name']!=self.expId : continue
            nCyc+=1
            #print('cc',nCyc,self.expId,expD['counts'])
            yL=[] # fixed order of labels
            self.Y.append(expD['counts'])  # will be 2D: [cyc][label]
        #print('unp',jid6, len(self.Y),len(yL2),'nCyc=',nCyc)
        #if self.prepState==0: print('unp np',np.array(self.Y).shape)
        assert nCyc>0
        
#...!...!....................
    def singleBit_counts(self,verb=1):
        self.Y=np.array(self.Y)  # make it numpy
        assert self.Y.ndim==2
        assert self.Y.shape[1]==2
        #print('qq',self.Y.shape)
        cnt={}
        for m in [0,1]:
            yV=self.Y[:,m]
            #print(m,yV.shape,yV)
            (avr,std,num,err)=npAr_avr_and_err(yV)
            cnt[m]=(avr,std,num,err)
        #print('aacnt',cnt)
        return cnt


#...!...!....................
    def XXsingleBit_counts_multiBitData(self,measBit,verb=1):
        self.Y=np.array(self.Y)  # make it numpy
        assert self.Y.ndim==2
        measInt= 1<<measBit   

        measIdx={}
        measIdx[1]=self.measLab[ self.measLab & measInt >0]
        measIdx[0]=self.measLab[ self.measLab & measInt ==0]
        #print('measIdx1=',measIdx[1])

        cnt={}
        for m in measIdx:
                yV=np.sum(self.Y[:,measIdx[m]],axis=1)
                #print(m,yV.shape,yV)
                (avr,std,num,err)=npAr_avr_and_err(yV)
                cnt[m]=(avr,std,num,err)
        print('aacnt',cnt)
        return cnt

