#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Part 3) retrieve  readout callibration for a device based on calib job result

Required INPUT:
   --job_id: 5d67...09dd  OR --jid6 j33ab44
   --transpName : ghz_5qm.v10

Reads matching  transp-file, recovers backend name
Computes measurement error mitigation matrix

Output: calibration + meta data
 out/mmitig.ghz_5qm.v10.yaml

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint

from NoiseStudy_Util import read_submitInfo
sys.path.append(os.path.abspath("../../utils/"))
from Circ_Plotter import Circ_Plotter
from Circ_Util import access_backend, circuit_summary, read_yaml, print_measFitter,  save_measFitter_Yaml, restore_measFitter_Yaml, retrieve_job

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-d',"--dataPath",default='hwjobs',help="input for everything")
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument('-J',"--job_id",default=None ,help="full Qiskit job ID")
    parser.add_argument('-j',"--jid6",default=None ,help="shortened job ID")
    parser.add_argument('-t','--transpName', default='hidStr_2qm.ibmq_boeblingen-v10',
                        help=" transpiled circuit which was  executed")


    args = parser.parse_args()
    args.prjName='mmitg.'+args.transpName
    
    args.dataPath+='/'
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!....................
def verify_calibJob_consistency( circD,jobHeadD,args):
    #print('verify_consistency circ:',circD['info']['name'],', jobHead:',jobHeadD)
    print('verify meas_qubit',str(args.metaD['expInfo']['meas_qubit']), str( circD['info']['meas_qubit']))
    assert str(args.metaD['expInfo']['meas_qubit'])== str( circD['info']['meas_qubit'])
    assert args.metaD['expInfo']['baseCircMD5']==circD['md5']
    print('check job vs. circ  passed')

#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
read_submitInfo(args)
args.prjName2=args.prjName+'-'+args.jid6

plot=Circ_Plotter(args )

# Import Qiskit classes here- I want to controll matlpot lib from Circ_Plotter(.)
from qiskit.ignis.mitigation.measurement import  CompleteMeasFitter

transpF=args.dataPath+'transp.'+args.transpName+'.yaml'
blobD=read_yaml(transpF)
if args.verb>2:
    print('M:  transp blob'); pprint(blobD)

circD=blobD['circOpt']
if args.verb>1:
    print('M: circOpt data:'); pprint(circD)
nqubit=circD['info']['qubits']
qubit_list=circD['info']['meas_qubit']
nclbit=len(qubit_list)
print('qubit_list=',qubit_list,',nclbit=',nclbit)

backName=args.metaD['submInfo']['backName']

state_labels=[] # regenarte  them
for i in range(2**nclbit):
    bitStr=bin(i)[2:].zfill(nclbit)
    state_labels.append(bitStr)
print('state_labels:',state_labels)

backend=access_backend(backName)
job, jobHeadD, jobResD = retrieve_job(backend, args.job_id, verb=args.verb)
verify_calibJob_consistency(circD,jobHeadD,args)

#
print('\n  - - - - - -  Analyzing the Results ',jobHeadD['date'])
cal_results = job.result()
# Calculate the calibration matrix with the noise model
meas_fitter = CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='jan23')
print_measFitter(meas_fitter, mxDim=3)

# save calibration results to file
auxD={}
for key in [ 'backend_name', 'job_id', 'time_taken']:
    auxD[key]=jobHeadD[key]
exp0=jobResD[0]
auxD['shots']=exp0['shots']
auxD['exec_date']=jobHeadD['date']
auxD['qubit_list']=qubit_list
auxD['nqubit_hw']=backend.configuration().n_qubits
auxD['circOpt']={'md5': circD['md5'], 'name':circD['info']['name']}
auxD['name']=args.prjName2

mmitgF=args.dataPath+args.prjName2+'.yaml'
save_measFitter_Yaml(meas_fitter,mmitgF,outD={'calibInfo':auxD})

if args.verb>1:
    print('\n  verify measFitte  was saved correctly') 
    meas_fitter3=restore_measFitter_Yaml(rdcalF)
    print('M: test, after restore:')
    print_measFitter(meas_fitter3, mxDim=3)

    
# Plot the calibration matrix
# self._tens_fitt.plot_calibration(0, ax, show_plot)
ax=plot.blank_page(13)
meas_fitter.plot_calibration(ax=ax,show_plot=False)
tit='qubits: '+''.join(str(qubit_list))
tit2=backend.name()
tit3='avr meas fidel: %.3f'%meas_fitter.readout_fidelity()
tit4='exec '+auxD['exec_date']
ax.text(0.07,-0.05,tit4,color='b',transform=ax.transAxes)
ax.text(0.07,0.15,tit3,color='b',transform=ax.transAxes)
ax.text(0.07,0.1,tit,color='b',transform=ax.transAxes)
ax.text(0.07,0.05,tit2,color='b',transform=ax.transAxes)

ax.text(0.27,0.9,args.prjName2,color='b',transform=ax.transAxes)

ax.grid(True)
plot.display_all()   
