#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"


'''
Train binary classifier :
- select data set
- select encoding
- select Ansatz

Records meta-data 
HD5 arrays contain trained model and history

Dependence: PennyLane

'''
import sys,os,hashlib
import numpy as np
from pprint import pprint
from time import time, localtime
import numpy as cnp  # Use cnp (conventional numpy) for standard numpy operations
from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from QML_Dichotomy import Trainer_Dichotomy
from toolbox.Util_IOfunc import read_yaml
from sklearn.model_selection import train_test_split
from PlotterDichotomy import Plotter
import argparse

#...!...!..................
def get_parser(backName="default"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or 'env'")
    parser.add_argument("--expName",  default=None,help='(optional) default is random string')
    
    parser.add_argument("--inputData",  default='circle2d2c',help='[.h5] input data ')
    parser.add_argument('--conf', default='SU2_ansatz_v1', help='[.conf.yaml] optimizer configuration')

    # .... quantum circuit
    parser.add_argument("--numQubit", default=3, type=int, help='size of circuit')
    #parser.add_argument('-i','--numSample', default=4, type=int, help='num of images packed in to the job')
    
    # .... training hyperparms
    parser.add_argument('-k','--numStep',type=int,default=None, help="(optional) steps of optimizer")
   
    #.... plotting
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("-p", "--showPlots",  default=' c d t ', nargs='+',help="abc-string listing shown plots")

    args = parser.parse_args()
    if 'env'==args.basePath: args.basePath= os.environ['PennyLane_dataVault']
    args.dataPath=os.path.join(args.basePath,'input')
    args.outPath=os.path.join(args.basePath,'model')
            
    for arg in vars(args):
        print( 'myArgs:',arg, getattr(args, arg))

    assert os.path.exists(args.outPath) 
    return args

#...!...!..................
def split_domains(X, Y, frac_train=0.7, frac_valid=0.2, minSampl=5, random_state=None):
    # Since frac_test is not directly provided, calculate it as the remaining fraction
    frac_test = 1 - frac_train - frac_valid
    
    # Ensure fractions sum to 1
    assert frac_train + frac_valid + frac_test == 1, "Fractions must sum to 1"

    # First split: separate out the test set
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=frac_test, random_state= random_state)

    # Adjust frac_valid to reflect the new total after removing the test set
    # The new fraction is calculated over the reduced set (1 - frac_test)
    frac_valid_adjusted = frac_valid / (1 - frac_test)

    # Second split: split the remaining data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=frac_valid_adjusted, random_state=random_state)

    # Assert that all subsets have at least M samples
    assert len(X_train) >= minSampl and len(X_val) >= minSampl and len(X_test) >= minSampl, f"All subsets must have at least {minSampl} samples."

    print('split_domains train/val/test X:',X_train.shape,X_val.shape,X_test.shape)
    if 0:
        print('train labels sample',Y_train[:10])
        print('train X sample',X_train[:3])
        print('val labels sample',Y_val)

    return (X_train,Y_train, X_val, Y_val,  X_test, Y_test)

#...!...!..................
def load_data(args,md):
    inpF=args.inputData+'.h5'
    bigD,md1=read4_data_hdf5(os.path.join(args.dataPath,inpF))

    md['data']=md1    
    X=bigD['data_X']
    Y=bigD['data_Y']
 
    print('\nLoad_data: X,Y sh:', X.shape, Y.shape)    
    print('Labels inp sample',Y[::30])

    frac_train=0.8; frac_valid=0.1
    XY_TVT=split_domains(X, Y,frac_train, frac_valid, random_state=43)

    #... add infor to MD
    tmd=md['train']
    tmd['split_dom_tvt']=[frac_train, frac_valid,1-frac_train-frac_valid]
    tmd['tot_sampl']=Y.shape[0]
    return XY_TVT

#...!...!....................
def buildTrainMeta(args):
    #... ingest training config 
    ocf=read_yaml(os.path.join('opt_conf',args.conf+'.conf.yaml'))    
    #1pprint(ocf)
    ansatzN=ocf['ansatz_name']
    assert ansatzN in ['CPhase','EffiSU2']
 
    cmd={}  # .... circuit
    cmd['num_qubit']=args.numQubit

    if ansatzN=='CPhase':
        cmd['param_shape']=[ocf['ansatz_layers'], 2*cmd['num_qubit']-1]

    if ansatzN=='EffiSU2':
        nUang=3
        cmd['param_shape']=[ocf['ansatz_layers'],cmd['num_qubit'],nUang]
        
    tmd={} #.... traning
    # ... some of those can be overwritten by args
    tmd['num_step']=ocf['max_steps']
    tmd['batch_size']=ocf['batch_size']
    assert args.numStep==None  # tmp
    
    md={'circuit':cmd,'train':tmd,'opt_conf':ocf}    
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    md['hash']=myHN
    if args.expName==None:
        name='qml_'+md['hash']
        md['short_name']=name
    else:
        md['short_name']=args.expName
    return md

#...!...!....................
def measure_expval_contour(args,trainer,bigD,md):
    print('measure_expval_contour...')
    # make data for decision regions
    xx, yy = np.meshgrid(np.linspace(-1,1, 30), np.linspace(-1,1, 30))
    X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

    X_data = np.array([x for x in X_grid])
    #print('X_data:',X_data.shape,len(X_grid))

    params=bigD['best_weights']
    #print('pp3',params.shape,type(params))
    
    pred_expval = trainer.infere(params, X_data)
    Z = np.reshape(pred_expval, xx.shape)
    bigD['pred_contour_x0_bins']=xx
    bigD['pred_contour_x1_bins']=yy
    bigD['pred_contour_expval']=cnp.array(Z)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser(backName='ibmq_qasm_simulator')
         
    jobMD=buildTrainMeta(args)
    XY_TVT=load_data(args,jobMD)    
    pprint(jobMD)
    
    trainer = Trainer_Dichotomy(jobMD,XY_TVT)
    trainer.train()
    
    # ....collect data
    trainer.summary()
    trainer.predict_test()
    jobD=trainer.bigD

    measure_expval_contour(args,trainer,jobD,jobMD)
    if args.verb>1: pprint(jobMD)
    
    
    #...... WRITE  OUTPUT .........
    outF=os.path.join(args.outPath,jobMD['short_name']+'.qmlModel.h5')
    write4_data_hdf5(jobD,outF,jobMD,verb=1)

    #print('   ./retrieve_ibmq_backRun.py --expName   %s   \n'%(expMD['short_name'] ))
   
    #--------------------------------
    # ....  plotting ........
    args.prjName=jobMD['short_name']
    #expMD['plot']={'addr_index':args.addrIndex,

    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.input_data('val',jobMD,jobD,figId=1)

    if 'b' in args.showPlots:
        plot.classified_data('val',jobMD,jobD,figId=2)        

    if 'c' in args.showPlots:
        ax,ax2=plot.expval_contour(jobMD,jobD,figId=3)
        plot.classified_data('val',jobMD,jobD,ax=ax)        
        plot.input_data('train',jobMD,jobD,ax=ax2)

    if 'd' in args.showPlots:
        ax,ax2=plot.expval_contour(jobMD,jobD,figId=4)
        plot.classified_data('test',jobMD,jobD,ax=ax)
        plot.input_data('train',jobMD,jobD,ax=ax2)        

    if 't' in args.showPlots:
        ax=plot.training_loss(trainer,figId=0)
 
    plot.display_all()
    print('M:done')
    pprint(jobMD['train'])
    #pprint(jobMD) #tmp
