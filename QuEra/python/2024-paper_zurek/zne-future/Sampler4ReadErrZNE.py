
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


import numpy as np
from pprint import pprint
from bitstring import BitArray
from toolbox.UAwsQuEra_job import flatten_ranked_hexpatt_array, build_ranked_hexpatt_array
from Util_zurek import accumulate_wallDens_histo
from collections import Counter

#............................
#............................
#............................

class Sampler4ReadErrZNE():
#...!...!....................
    def __init__(self,expD, meta, args):
        self.verb=args.verb
        self.expD=expD
        self.meta=meta

        pmd=meta['payload']
                
        args.noiseScale.sort()
        assert len(args.noiseScale)>=2
        assert args.noiseScale[0]>1.01

        zmd={}
        zmd['noise_scale']=args.noiseScale
        zmd['sample_reuse']=args.sampleReuse
        zmd['readout_error']=args.readoutError
        meta['zne_wall_dens']=zmd

        print('ReadErrZNE_wallDens():')
        pprint(zmd)
        noiseLambdaV=np.array(zmd['noise_scale'])
        read01noise=np.array(zmd['readout_error']) # error: [ s0m1, s1m0]
        self.read01fidelity=1-read01noise
        nLambda=len(zmd['noise_scale'])+1  # the input is added as the firts
        nAtom=meta['postproc']['num_atom']
        recExt=''
        
        isIdealSample='ideal_sample' in pmd
        if isIdealSample:
            print('ReZNE: detected "ideal_sample", apply readout error first')
            recExt='_ideal'
            nLambda+=1

        if  args.onlySPAM: nLambda=2 # for Milo to not run ZNE, tmp
        
        # final storage
        self.wallDens2D=['-' for i in range(nLambda) ] # distr
        self.densMomV=['-' for i in range(nLambda) ] # avr+std moments of density
        self.noiseScale=['-' for i in range(nLambda) ] # as applied to data
        self.iLambda=0

        bitpattL=self._flatten_ranked_bitpatt(recExt)  # unpack the input
        
        if isIdealSample:
            self.noiseScale[self.iLambda]=0. # no SPAM
        else:  # input has SPAM already, the case of  real data
            self.noiseScale[self.iLambda]=1.  # 1 x SPAM
            self.meas_bitpattL=bitpattL
        
        self.iLambda+=1
        if not isIdealSample: # started from SPAM'ed data
            return
        
        #.... apply  SPAM noise for 'ideal' input
        self.meas_bitpattL=self._amplifyReadoutError(bitpattL,self.read01fidelity)
        
        solCounter=Counter(self.meas_bitpattL)  #.... convert list to dict
        countsV,hexpattV,hammV,nAtom2=build_ranked_hexpatt_array(solCounter)
        self.expD['ranked_counts']=countsV
        self.expD['ranked_hexpatt']=hexpattV
        wallDensV,wallMoments=accumulate_wallDens_histo(self.meas_bitpattL,isLoop=nAtom==58)
        self.noiseScale[self.iLambda]=1.  # 1 x SPAM
        self.wallDens2D[self.iLambda]=wallDensV
        self.densMomV[self.iLambda]=wallMoments # [ mean_x, mean_err, std_x, std_std, tshot]
        self.iLambda+=1
            
        #print('tmp1',sorted(expD))
        return
        
 
#...!...!....................
    def finalize(self):
        print('SAMfin: iLambda=%d of %d'%(self.iLambda,len(self.densMomV)))

        self.expD["SPAM_scale"]=np.array(self.noiseScale)
        self.expD["domain_wall_dens_histo"]=np.array(self.wallDens2D)
        self.expD["domain_wall_dens_moments"]=np.array(self.densMomV)
        print('SAMfin:expD',sorted(self.expD))

#...!...!....................
    def amplify_SPAM(self):
        zmd=self.meta['zne_wall_dens']
        nAtom=self.meta['postproc']['num_atom']
        noiseV=zmd['noise_scale']
        nRepeat=zmd['sample_reuse']
        print('aa NSC:', noiseV)
        nScale=len(noiseV)
        
        for js in range(nScale):
            scale=noiseV[js]
            redFid=self.read01fidelity**(scale-1)
            print('\n******\ndo noise scale=%.2f, redFid:'%scale,redFid)
            bitpattL=self._amplifyReadoutError(self.meas_bitpattL,redFid,nRepeat=nRepeat)
            wallDensV,wallMoments=accumulate_wallDens_histo(bitpattL,isLoop=nAtom==58)
            self.noiseScale[self.iLambda]=scale
            self.wallDens2D[self.iLambda]=wallDensV
            self.densMomV[self.iLambda]=wallMoments # [ mean_x, mean_err, std_x, std_std, tshot]
            self.iLambda+=1
            
        
#...!...!....................
    def _flatten_ranked_bitpatt(self,nameExt):
        # permanently alter the bigData
        countsV=self.expD['ranked_counts'+nameExt] 
        hexpattV=self.expD['ranked_hexpatt'+nameExt] 

        ppmd=self.meta['postproc']
        nAtom=ppmd['num_atom']
        print('flatten %d used shots'%(ppmd['used_shots']))
        bitpattL=flatten_ranked_hexpatt_array(countsV,hexpattV,nAtom)        
        print('dump bitpattL:', len(bitpattL), bitpattL[:2])
        assert len(bitpattL)==ppmd['used_shots']

        wallDensV,wallMoments=accumulate_wallDens_histo(bitpattL,isLoop=nAtom==58)
        self.wallDens2D[self.iLambda]=wallDensV
        self.densMomV[self.iLambda]=wallMoments # [ mean_x, mean_err, std_x, std_std, tshot]
        
        return bitpattL
         
#...!...!....................
    def _amplifyReadoutError(self,inp_bitstrL,readFid,nRepeat=1):
        nShot=len(inp_bitstrL)
        print('ARE:readFid %s, shots=%d, nRepeat=%d'%(readFid,nShot,nRepeat))
        nAtom=len(inp_bitstrL[0])
              
        def apply_spam_distortion(patt, readFid):
            #print('DPT:inp',patt)
            outS=['-' for i in range(nAtom) ]  # empty output
            rndP=np.random.rand(nAtom)
            for i in  range(nAtom):
                x=int(patt[i])
                p=rndP[i]
                if p > readFid[x] :  x=1-x
                outS[i]=str(x)
            patt2=''.join(outS)
            #print('DPT:out',patt2)
            return patt2

        out_bitstrL=[ '=' for i in range(nShot*nRepeat) ]  # empty output
        j=0
        for k in range(nShot):
            bitstr=inp_bitstrL[k]
            for i in range(nRepeat):
                out_bitstrL[j]=apply_spam_distortion(bitstr, readFid)
                j+=1

        return out_bitstrL

#...!...!....................
    def XX_ammplifyReadoutError(self,noiseFull,zmd,inpL):

        nScale=noiseFull.shape[0]
        '''
        HW applied noiseScale=1 and measured bits have fidelity 'readFid'
        to achieve  lower fidelity= readFid^noiseFull we need to only degrade measured bitstrings with reduceFid = readFid^(noiseFull-1)        
        '''
        readFidV=1-np.array(zmd['readout_error']) # error: [ s0m1, s1m0]

        reduceFid=[]
        for noise in noiseFull:
            reduceFid.append( readFidV ** (noise-1.) )
        # print('reduceFid:',reduceFid)

       
        outLL=[ [] for i in range(nScale) ]
        
        for binpatt in inpL:
            #binpatt='10'*27
            for js in range(nScale):
                out=[]
                redFid=reduceFid[js]
                for jm in range(zmd['sample_reuse']):
                    pattx=distort_pattern(binpatt, redFid)
                    out.append( pattx)
                outLL[js]+=out
        for js in range(nScale):
            print('dump outLL[%d]:'%js, len(outLL[js]), outLL[js][:1])

        return  outLL
