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
        yields=blob.pop('yieldsMitg')
        metaD=blob
        print('metaD:', metaD.keys())
        print('see %d experiments'%len(yields))
        return metaD,yields


#............................
#............................
#............................
class OneExp_Ana(object):
    def __init__(self, expId,labels,metaD):
        self.expId=expId
        self.labels=labels
        self.jid6L=[]  # keep track of all jobs
        self.execT=[]
        self.Y=[]
        self.shots=metaD['expInfo']['shots']
        self.metaD=metaD
        
#...!...!....................
    def unpack_yieldArray(self,jid6, execDate, yields):
        self.jid6L.append(jid6)
        self.execT.append(execDate)
        #print('ny',len(yields))
        # note, same expId is present mutiple times
        nCyc=0
        yL2=[]
        for expD in yields:
            if expD['injGate']!=self.expId : continue
            nCyc+=1
            #print('cc',nCyc,expD['injGate'])
            assert self.shots==expD['shots']
            yL=[] # fixed order of labels
            #print(expD['counts'].keys())
            for lab in  self.labels:
                val=0
                if lab in expD['counts']: val=expD['counts'][lab]
                yL.append(val)  # always all labels are on the list
            yL2.append(yL)
        self.Y.append(yL2)  # will be 3D: [jid][cyc][label]
        #print('unp',jid6, len(self.Y),len(yL2),'nCyc=',nCyc)
        #print('as np',np.array(self.Y).shape)
            
#...!...!....................
    def analyzeIt_plain(self):
        #print('ana expId=',self.expId)
        self.Y=np.array(self.Y)  # make it numpy
        assert self.Y.ndim==3
       
        probL=[]; infoL=[]; probA=[]; infoA=[]; 
        for iLab,lab in enumerate(self.labels):
            Y=self.Y[:,:,iLab].flatten()
            # inflate to 1  yields <1. - to compensate the error mitigation 
            Y[Y<1.0]=1. # is this correct?
            num=Y.shape[0]
            Prob_lab=Y/self.shots  # prob(lab)
            Info_lab=-np.log(Prob_lab)
            
            probL.append(Prob_lab)  # probability  per experiment
            infoL.append(Info_lab) # information  per  experiment

            # avereged over set of experiments
            # entropy=np.sum( Info_lab)/num

            avrProb=npAr_avr_and_err(Prob_lab)
            avrInfo=npAr_avr_and_err(Info_lab)

            probA.append(avrProb)  # probability averaged over exp.
            infoA.append(avrInfo) # information averaged over exp. 
             
        self.dataV={'prob':probL, 'info':infoL, 'probAvr':probA,'infoAvr':infoA}
        # convert all to np-arr: [label][experiments]
        for k in self.dataV:
            ar=np.array(self.dataV[k])
            #print('ana-end key=',k,ar.shape)
            self.dataV[k]=ar
        
        
#...!...!....................
    def compensated_analysis(self,baseData_perJob=None,verb=0):
        #print('ana_comp expId=',self.expId)
        self.Y=np.array(self.Y)  # make it numpy
        assert self.Y.ndim==3

        if baseData_perJob!=None:
                [base_avrProb,_,_,_]=baseData_perJob['avrProb']
                [base_avrSelfInfo,_,_,_]=baseData_perJob['avrSelfInfo']
        
        Y=self.Y
        # operate directly on 3D npAr [jobs, cycle,label]
        Y[Y<1.0]=1. # is this correct? Jan is not sure here
        #print('EM2:',Y.shape)#,base_avrProb.shape)
        
        prob=Y/self.shots
        sinfo=-np.log(prob)
        
        #print('base',base_avrProb[0,1])
        #print('pre prob',prob[:,0,1])

        if baseData_perJob!=None:
            # compensate for job-2-job variation using base-circ data
            for iCyc in range(Y.shape[1]):
                prob[:,iCyc,:]/=base_avrProb
                sinfo[:,iCyc,:]-=base_avrSelfInfo
                
        #print('post prob',prob[:,0,1])

        probL=[]; sinfoL=[]; avrProbL=[]; avrSinfoL=[]; 
        for iLab in  range(Y.shape[2]):
            probL.append(prob[:,:,iLab])
            sinfoL.append(sinfo[:,:,iLab])
            
            # probability averaged over exp:
            avrProbT=npAr_avr_and_err(prob[:,:,iLab],axis=None)
            avrProbL.append(avrProbT)  

            avrSinfo=npAr_avr_and_err(sinfo[:,:,iLab],axis=None)
            avrSinfoL.append(avrSinfo) 
        
        probL=np.stack(probL)
        sinfoL=np.stack(sinfoL)
        #print('ss1',probL.shape)
        #print('ss2',infoA.shape,infoA[1,:])
        self.dataV={'prob':probL, 'selfInfo':sinfoL, 'avr_prob':avrProbL,'avr_selfInfo':avrSinfoL}
        
        
#...!...!....................
    def analyzeIt_perJob(self, verb=0):
        print('ana_perJob expId=',self.expId)
        self.Y=np.array(self.Y)  # make it numpy
        assert self.Y.ndim==3

        Y=self.Y
        # operate directly on 3D npAr [jobs, cycle,label]
        Y[Y<1.0]=1. # is this correct? Jan is not sure here
        #print('EM1:',Y.shape)
        
        prob=Y/self.shots 
        selfInfo=-np.log(prob)
        # avereged over set of experiments
        # entropy=np.sum( Info)/num
        
        avrProb=npAr_avr_and_err(prob,axis=1)
        avrSelfInfo=npAr_avr_and_err(selfInfo,axis=1)

        #print('ss',prob.shape,len(avrProb),avrProb[0].shape )
        self.data_perJob={'prob':prob, 'selfInfo':selfInfo, 'avrProb':avrProb,'avrSelfInfo':avrSelfInfo}
        return
      
        
