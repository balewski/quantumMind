#!/usr/bin/env python3
'''
automatic optimizer of Hamitonian for MIS problem

./autoOptRabiV2.py -E autoRabiV2 

XX
Summary of results is at:
data/auto2Q6Meas_3120655e.progress.txt   (1 line per experiment)

You can display best result w/ 
./fit_gaussMix1Q.py  -M  auto2Q6Meas_3120655e_43 -X
a_ auto2Q6Meas_3120655e_43 -G same

'''

import sys,os
#sys.path.append(os.path.abspath("toolbox/"))
from AutoOptHamiltonianMIS import AutoOptHamiltonianMIS

import time
import numpy as np
import copy
from lmfit import Minimizer, Parameters, fit_report
from pprint import pprint
from toolbox.Util_miscIO import read_yaml
class EmptyClass:  pass
from toolbox.UAwsQuEra_job import submit_args_parser
import argparse

#...!...!..................
def get_parser(backName="cpu"):  # add task-speciffic args
    parser = argparse.ArgumentParser()

    #  ProblemMIS task speciffic
    parser.add_argument('--atom_dist_um', default='4.1', type=str, help='distance between 2 atoms, in um')
    parser.add_argument('--grid_shape', default=['square',4],  nargs='+', help='grid x & y size, space separated list, OR:   square & x-size ')
    
    parser.add_argument('-s','--grid_seed', default=92, type=int, help='seed for graph generator, 0=none')



    parser.add_argument('--grid_droput', default=0.45, type=float, help='fraction of removed atoms from the perfect grid')


    args=submit_args_parser(backName,parser,verb=0)
    return args


#...!...!..................
def pre_configure(args):
    
    # remove not used arguments
    
    del args.fakeSubmit
    
    # add needed arguments
    args.outPath='out'  #tmp
    #args.confPath='auto_conf'
    args.noXterm=True # do not want plots to pop-up
    args.multi_clust=False # too complicated for now
 
    #... ingest config
    tc=read_yaml(os.path.join('auto_conf','autoHamilMIS_v1.conf.yaml'))
    pprint(tc)
    args.taskConf=tc
    
    # count number of detune params
    nDetune=len([ x for x in tc['start'] if 'detune_a' in x ])
    args.num_detune_par=nDetune  # ugly solution
    #... append more config params
    #args.detune_shape=tc['detune_shape']
    args.detune_shape=[ tc['start']['detune_a%d'%i] for i in range(nDetune) ]
    args.atom_dist_um=tc['atom_dist_um']
    args.evol_time_us=tc['evol_time_us']
    
    if args.expName==None: args.expName=tc['exp_name']

    print("M: modiffied arguments")
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
   
    #...  expand config
    varyD=tc['vary']
    for pname in varyD:
        pvary,pmin,pmax=varyD[pname]
        pdel=pmax-pmin
        print('Range pname=',pname,pmin,pdel,pvary) 
        assert  pdel>0
        varyD[pname].append(pdel)
        
#...!...!..................
def prime_start(tc):
    # create a set of Parameters
    uparams = Parameters()  # passed to minimizer
    varyD=tc['vary']
    for pname in varyD:
        pval=tc['start'][pname]
        pvary,pmin,_,pdel=varyD[pname]
        uval=(pval-pmin)/pdel
        uname='U'+pname  # prepend U
        print('pname=%s pval=%.3f uval=%.3f'%(pname,pval,uval))        
        uparams.add(uname, value=uval,min=0,max=1., vary=pvary)
    print('M:initilized uparams')
    uparams.pretty_print()
    return uparams



#...!...!..................
def MIS_loss(expD,md): # loss for Max-indep-set
    amd=md['analyzis']
    nAtom=amd['num_atom']
    probV=expD['ranked_probs'] # [PEY,NX,NB]
    MISs=expD['ranked_MISs']   # [NX,IdCa]

    if 0: # very simple:  just count fraction of 0-states
        nG=0; nAny=0
        for mshot,card in zip(probV[2,:,0],MISs[:,1]):
            #print('mshot,card;',mshot,card)
            nG+=mshot*card  
            nAny+=mshot*nAtom
        probG=nG/nAny
        print('nG=%.2e  nAny=%.2e  probG=%.3f'%(nG,nAny,probG))
        return probG

    if 1: # select a very scpeciffic state: 0-3-6-9 fro sqyare-circ 4x4
        for i,rec in enumerate(MISs):
            #print('i',i,rec)
            if rec[0]==2340:  # 0-3-6-9
            #if rec[0]==1170: #  1-4-7-10
                print('found golden state, rank=',i)
                print('probs:',probV[:,i,1])#,probV[:,i,0])
                return np.sqrt(probV[0,i,0]) # because loss will be L2
        return 1.0

#...!...!..................
def fcn2min(uparams): # function to be minimized
    nDetune=exper.num_detune_par
    #print(tc)
    varyD=tc['vary']
    tmD=tc['time_match']  # for warm-up????
    
    # COBYLA changes 2 frequencies
    inpD={} # passed to experiment
    for uname in uparams:
        uval=uparams[uname].value
        #print('fcn2min-Uval:', uname, uval, uparams[uname].vary)
        pname=uname[1:] # skip U
        pvary,pmin,_,pdel=varyD[pname]
        # update values , for not-varied it works as well           
        inpD[pname]=float( uval*pdel+pmin)
        #print('fcn2min-par pname:', pname, inpD[pname],'upar:',uval, uparams[uname].vary)
            
    
    # EXECUTE EXPERIMENT ....
    exper.run_one_experiment(inpD)
    loss1=MIS_loss(exper.expD,exper.meta) # loss for Max-indep-set

    # wrap up 
    resV=np.array([loss1])
   
    #print('ggg1',resV) #   print('ggg2',resV**2)
    xx=np.sum( resV**2)
    short_name=exper.getShortName()
    elaT=time.time()-startT
    #print('M:resV=',resV)
    myIter=exper.cnt['run']-1
    txtDetu=' '.join('%.2f'%(inpD['detune_a%d'%i]) for i in range(nDetune) )
    txt='%s it%d loss= %.3f  ramp_us up=%.2f down=%.2f detune: %s, elaT=%.1f min'%(short_name,myIter, xx,inpD['ramp_up_us'],inpD['ramp_down_us'],txtDetu,elaT/60.)

    print('M:exper: ' +txt)
    print('M:input'); pprint(inpD)
    fds.write(txt+'\n')

    exper.totIter=myIter
    if exper.bestLoss>xx:
        exper.bestLoss=xx
        exper.bestInpD=copy.copy(inpD)
        exper.bestExper=short_name
        exper.bestIter=myIter
        exper.bestSum=txt        
        #print('ddd');pprint(exper.bestInpD)
    return resV


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    startT=time.time()
    args=get_parser()
    pre_configure(args)

    exper=AutoOptHamiltonianMIS(args)
    
    outF=exper.getShortName()+'.progress.txt'
    print('M:open',outF)
    fds=open(os.path.join(args.outPath,outF),'w', buffering=1) # flush per line
    exper.bestLoss=9e99
    exper.bestInpD={}
    
    tc=args.taskConf
    uparams =prime_start(tc)
    
    ####################################################################

    #1exper.run_one_experiment(tc['start']) # for testing, run central value experiment
    
    #tc['max_experiment']=1
    print('\nM:initialize Minimizer:')
    fitter = Minimizer(fcn2min, uparams,max_nfev=tc['max_experiment'])
    
    #opt_method='leastsq' #???
    opt_method='cobyla' # works for QubiC but needs 40-60 exepriments to converge
    trial = fitter.minimize(method=opt_method)
    print(fit_report(trial))
    

    fds.write('auto2-done numExperiment=%d  elaT=%.1f min\n'%(exper.cnt['run'],(time.time() - startT)/60.))
    if len(exper.bestInpD)>0:
        fds.write('\nbestExper %s MSE-loss=%.3f iter=%d of %d \n'%(exper.bestExper,exper.bestLoss,exper.bestIter,exper.totIter))
        #print( '\nM: bestExper %s MSE-loss=%.1f'%(exper.bestExper,exper.bestLoss))
        print( 'M: bestSum '+exper.bestSum)
        for x in exper.bestInpD:
            valTxt='%.3f'%exper.bestInpD[x]
            #if 'freq' in x:  valTxt='%.3fe6'%(exper.bestInpD[x]/1e6)
            fds.write('    %s: %s\n'%(x,valTxt))
            
        fds.write('sum: %s\n'%exper.bestSum)

    print('\nM:auto-done numExperiment=%d  elaT=%.1f min'%(exper.cnt['run'],(time.time() - startT)/60.))
    
    
 

         
    

    
