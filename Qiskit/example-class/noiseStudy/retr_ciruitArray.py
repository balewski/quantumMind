#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''

Part 5) retrieve  target circuit results and correct the yileds baswed on  previously measured readout callibration

Required INPUT:
   --job_id: 5d67...09dd  OR --jid6 j33ab44
   --transpName : ghz_5qm.v1
   --measCalib :  j3d0f8b

Reads matching  transp-file, recovers backend name
  data/transp.ghz_5qm.v10.yaml
Reads matching  measurement error mitigation matrix
  data/mmitig.ghz_5qm.v10.yaml

Output:  raw & corrected yields + meta data TBD


'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint

from qiskit.ignis.mitigation.measurement import  CompleteMeasFitter

from NoiseStudy_Util import read_submitInfo
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Plotter import Circ_Plotter
from Circ_Util import access_backend, circuit_summary, write_yaml, read_yaml, print_measFitter,  save_measFitter_Yaml, restore_measFitter_Yaml, retrieve_job

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for calib+circ")
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-c','--measCalib', default='j-fixMe',
                        help=" id of calibration job, or none to skip")

    parser.add_argument('-J',"--job_id",default=None ,help="full Qiskit job ID")
    parser.add_argument('-j',"--jid6",default='je840f5' ,help="shortened job ID")
    parser.add_argument('-t','--transpName', default='hidStr_2qm.ibmq_boeblingen-v10',
                        help=" transpiled circuit to be executed")


    args = parser.parse_args()
    args.prjName='measCirc_'+args.transpName
    args.calibName='mmitg.'+args.transpName+'-'+args.measCalib

    args.dataPath+='/' 
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def yields_2_metaD(args,jobResMitgD,expMetaD):
    outL=[]
    for exp in jobResMitgD:
        #print('gg'); pprint(exp); ok34
        counts=exp['data']['counts']
        counts2={ x:float(counts[x] ) for x in counts} # get rid of numpy
        expName=exp['header']['name']
        injGate=expMetaD[expName]['injGate']
        outL.append( { 'shots': exp['shots'], 'counts':counts2  , 'name':expName,'injGate':injGate})
    args.metaD['yieldsMitg']=outL
    #print('rrr'); pprint(outL)

#...!...!....................
def verify_circArr_consistency(calibMetaD, circD,jobHeadD):
    print('verify_consistency circ:',circD['info']['name'],', calib:',calibMetaD['name'])
    assert circD['md5']==calibMetaD['circOpt']['md5']
    assert str(calibMetaD['qubit_list'])== str( circD['info']['meas_qubit'])
    print(calibMetaD['backend_name'], jobHeadD['backend_name'])
    assert calibMetaD['backend_name']==jobHeadD['backend_name']
    print('check calib vs. circ  passed')

    
#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
read_submitInfo(args)

transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
blobD=read_yaml(transpF)
if args.verb>2:
    print('M:  transp blob'); pprint(blobD)
circD=blobD['circOpt']
if args.verb>1:
    print('M: circOpt data:'); pprint(circD)
    
if args.measCalib!='none':
    # load readout calibration
    mmitgF=args.dataPath+args.calibName+'.yaml'
    meas_fitter, calibMetaD=restore_measFitter_Yaml(mmitgF)
    print('M: after restore:')
    print_measFitter(meas_fitter, mxDim=2)
else:
    print('Warn, read-error mitigation not performed')
    
          
backName=args.metaD['submInfo']['backName']
backend=access_backend(backName)
job, jobHeadD, jobResD = retrieve_job(backend, args.job_id, verb=args.verb)

numExp=len(jobResD)

if args.measCalib!='none':
    verify_circArr_consistency(calibMetaD, circD,jobHeadD)
    # Get the filter object
    meas_filter = meas_fitter.filter

    jid6='j'+args.job_id[-6:]
    print('M: apply read-err-mitg for %d experiments ...'%(numExp))
    startT = time.time()
    jobMitgD = meas_filter.apply(job.result()).to_dict()
    assert jobMitgD['success']==True
    jobResMitgD=jobMitgD.pop('results')
    print('read-err-mitg done, took %d sec, jid6=%s'%(time.time()-startT,jid6))
else:
    jobMitgD =job.result().to_dict()
    jobResMitgD=jobMitgD.pop('results')
    calibMetaD='none'
    
yields_2_metaD(args,jobResMitgD,args.metaD['expInfo']['expMeta'])
# add more from retrieve here

args.metaD['retrInfo']={'baseCircMD5':circD['md5'], 'time_taken':jobHeadD['time_taken'],'exec_date':jobHeadD['date'] } 
args.metaD['calibInfo']=calibMetaD

outF=args.dataPath+'/yieldArray-%s.yaml'%args.jid6
write_yaml(args.metaD,outF)

for iexp in range(numExp):
    mitg_counts = jobResMitgD[iexp]['data']['counts']
    #raw_counts = jobResD[iexp]['data']['counts'] # gives state='0x3'
    raw_counts = job.result().get_counts(iexp) # gives state='11'
    print('iexp=',iexp,'shots=',args.metaD['expInfo']['shots'])
    print('raw',raw_counts)
    print('mitg',mitg_counts)

    break

print('End-OK')
