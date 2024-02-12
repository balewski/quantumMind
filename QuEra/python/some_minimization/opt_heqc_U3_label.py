#!/usr/bin/env python3
'''
automatic optimizer of Hamitonian for Z2Phase problem

'''

import sys,os
from AutoOpt_heqc_U3_label import AutoOpt_heqc_U3_label

import time
import numpy as np
import copy
from lmfit import Minimizer, Parameters, fit_report
from pprint import pprint
from Util_miscIO import read_yaml
class EmptyClass:  pass
import argparse

# for 2pt-corr loss
#from ProblemZ2Phase import ProblemZ2Phase

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3],  help="increase output verbosity", default=1, dest='verb')

    parser.add_argument('--conf', default='u3_label_v1', help='[.conf.yaml] optimizer configuration')
    
    parser.add_argument("--nqData",  default=3, type=int, help='size of value registers')
        
    parser.add_argument( "-B","--noBarrier", action='store_true', default=False, help="remove all bariers from the circuit ")
    
    parser.add_argument('-n','--numShots',type=int,default=40000, help="shots per QBart address, if negative use as-is ")
    parser.add_argument( "-F","--fakeJob", action='store_true', default=False,help="do not minimize, run circuit once")
    
    args = parser.parse_args()
    # make arguments  more flexible
    args.numSample=1
    args.expName=None
    args.nqLabel=args.nqData  # size of integer label
    args.oracleType='realOverlap'  # see 2023 IEEE QCAM paper fig 6b
    args.decompLevel=0
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    return args


#...!...!..................
def pre_configure(args):    
    # remove not used arguments    
       
    # add needed arguments
    args.outPath='out'  #tmp
    args.short_name=args.conf
    #... ingest config  to modify args
    tc=read_yaml(os.path.join('opt_conf',args.conf+'.conf.yaml'))
    tcs=tc['start']
    pprint(tc)
    args.taskConf=tc
    # clenup varied params
    for pname in tc['vary']:
        rec=tc['vary'][pname]
        pvary=rec[0]
        if pvary=='F': rec[0]=False
        if pvary=='T': rec[0]=True
    
    # count number of detune params
    nDetune=len([ x for x in tcs if 'detune_a' in x ])
    args.num_detune_par=nDetune  #
    
    #... append more config params to args 
    args.detune_shape=[ tc['start']['detune_a%d'%i] for i in range(nDetune) ]
 
    #args.atom_dist_um=tc['atom_dist_um']
    args.evol_time_us=tc['evol_time_us']
    args.numShots=tc['num_shots']
    args.rabi_omega_MHz=tc['rabi_omega_MHz']
    args.detune_delta_MHz=tc['detune_delta_MHz']

    if args.expName==None: args.expName=tc['exp_name']
    args.atom_dist_um=-1  # updated in run_one_experiment(.)
   
    print("M: modiffied arguments")
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))
   
    #...  expand config for optimizer
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
        print('pname=%s pval=%.3f uval=%.3f'%(pname,pval,uval),'vary:',pvary,type(pvary))        
        uparams.add(uname, value=uval,min=0,max=1., vary=pvary)
    print('M:initilized uparams')
    uparams.pretty_print()
    return uparams


#...!...!..................
def my_loss_MIS(expD,md): # loss for maximization of MIS probability
    #amd=md['analyzis']
    #nAtom=amd['num_atom']
    target_hex=md['payload']['target_state_hex']
    probV = expD['ranked_probs']       # [PEY,NB]
    hexpattV= expD['ranked_hexpatt']   #[NB]

    if 1: # select a very scpeciffic state: 1010...01
        for i,rec in enumerate(hexpattV):
            #print('i',i,rec)
            if rec==target_hex:
                print('found golden state:%s, rank=%d'%(target_hex,i))
                print('probs:',probV[:,i])
                return np.sqrt(1-probV[0,i]),i # because loss will be L2
        return 1.0,-1


    
#...!...!..................
def fcn2min(uparams): # function to be minimized
    nDetune=exper.num_detune_par
    #print(tc)
    varyD=tc['vary']
        
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
    if 1:
        loss1,rank1=my_loss_MIS(exper.expD,exper.meta) # loss for Max-indep-set
    else:
        loss1,rank1=my_loss_2ptCorr(exper.expD,exper.meta) # loss for 2pt-corr
        not_tested_and_not_working_yet
        
    # wrap up 
    resV=np.array([loss1])
    xx=np.sum( resV**2)
    short_name=exper.getShortName()
    elaT=time.time()-startT
    #print('M:resV=',resV)
    myIter=exper.cnt['run']-1
    txtDetu=' '.join('%.2f'%(inpD['detune_a%d'%i]) for i in range(nDetune) )
    txtTTT='ramps: %.2f %.2f'%(inpD['t_ramp_up_us'],inpD['t_ramp_down_us'])
    txtANC='anc: %.2f'%inpD['anc_xdist_um'] if varyD['anc_xdist_um'][0] else ''
    txt='%s it%d loss=%.3f %s shape: %s %s atom: %.2f rank:%d elaT=%.1f min'%(short_name,myIter, xx,txtTTT,txtDetu,txtANC,inpD['atom_dist_um'],rank1,elaT/60.)
    if exper.bestLoss>xx:   txt='*'+txt
    else:  txt=' '+txt
    print('M:input'); pprint(inpD)
    print('M:result ' +txt)
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
    # Set a different seed based on the current time
    np.random.seed(int(time.time()))
 
    startT=time.time()
    args=get_parser()
    pre_configure(args)

    exper=AutoOpt_heqc_U3_label(args)
    
    outF=args.short_name+'.progress.txt'
    print('M:open',outF)
    fds=open(os.path.join(args.outPath,outF),'w', buffering=1) # flush per line
    exper.bestLoss=9e99
    exper.bestInpD={}
    
    tc=args.taskConf
    uparams =prime_start(tc)
    
    ####################################################################

    if args.fakeJob:
        exper.run_one_experiment(tc['start'])
        exit(0) # for testing, run central value experiment
    
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
    
    
 

         
    

    
