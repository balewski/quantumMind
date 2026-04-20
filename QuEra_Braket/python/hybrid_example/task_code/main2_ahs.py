#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# executed simply AHS problem included

import os,sys
import json
from pprint import pprint
import socket  # for host name

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--numIter", default=5, help="number of iterations")
    parser.add_argument("--device_arn", default=None, help="Amazon Resource Name if Aqula is used")
    parser.add_argument("--backend_short", default="local", help="short backend name")
    parser.add_argument("--short_name", default=None, help="short job  name")
    parser.add_argument('-n','--numShots', default=26, type=int, help='num of shots')
    parser.add_argument("--outPath",default='out/',help="output paths")
    args = parser.parse_args()
    # transformations improving flexibility
    args.awsHybrid=False
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def inspect_dir(head,text='my123'):
    print('\n----Inspect %s  head=%s'%(text,head))
    try:
        directory_contents = os.listdir(head)
    except:
        print('head note exist, skip'); return
        
    # Print the list of contents
    for item in directory_contents:
        c='/' if os.path.isdir(os.path.join(head,item)) else ''        
        print(item+c)

#...!...!....................
def apply_hyperparams(args,hpar,outPath):
    print('\nAWS input hpar:'); pprint(hpar)
    args.awsHybrid=True
    args.numIter=int(hpar['num_iter']) # AWS passes only strings
    args.numShots=int(hpar['num_shots'])
    args.backend_short=hpar['backend_short']
    args.short_name=hpar['short_name']
    
    if hpar['backend_short']=='aquila':
        args.device_arn= hpar['quera_device_arn']
    args.outPath=outPath
    
    print('\n\nRevised args:')
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    print()
    
#...!...!....................
def inspect_AWS_dirs1():
    # ... inspect directories ....
    path=os.getcwd()
    inspect_dir(path, 'working dir')
    path+='/task_code'
    inspect_dir(path, 'task_code dir')
    path+='/toolbox'
    inspect_dir(path, 'toolbox dir')

#...!...!....................
def inspect_AWS_dirs2():
    # ... inspect directories based on AWS env ....
    envD=env_2_dict('SM_TRAINING_ENV',verb=0)

    pathOut=envD['output_data_dir']
    inspect_dir(pathOut,'pathOut')

    pathInp=envD['channel_input_dirs']['input']
    inspect_dir(pathInp,'pathinp')
    return envD,pathInp, pathOut


#...!...!....................
def execute_task(args,awsHpar):    
    
    print('I AM execute_task(), check awsHybrid:',args.awsHybrid)        
    print('  device_arn=%s backN=%s'%(args.device_arn,args.backend_short))
 
    if args.awsHybrid :
        #.... save config as JSON       
        outF=os.path.join(args.outPath,'ahs_task_conf.%s.json'%args.backend_short)
        with open(outF, 'w') as json_file:
            md={'name1':'ahsToy'}
            md['hyperparameters']= awsHpar
            json.dump(md, json_file)
        print('RHS: aws hpar saved at '+outF)

    run_hybrid_task(args)
 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    args=get_parser()
    hostname = socket.gethostname()
    print('I AM MAIN(), hostname:', hostname)
    # AWS:  hostname: ip-10-2-71-127.ec2.internal
    # podman image:  hostname: 8ca1120d30b6
    awsHybrid='ec2.internal' in hostname 

    awsHpar=None
    if awsHybrid:  # fail-safe code localization
        inspect_AWS_dirs1()
        codeDir=os.getcwd()+'/task_code'
        print('M: assume codeDir=',codeDir)
        sys.path.append(codeDir)

    # this should  work  locally & on AWS
    from toolbox.Util1 import env_2_dict,dump_file
    from toolbox.Util_AHS  import run_hybrid_task
    
    if awsHybrid:
        awsEnvD,pathInp, pathOut=inspect_AWS_dirs2()
        awsHpar=awsEnvD['hyperparameters']
        apply_hyperparams(args,awsHpar, pathOut)  

    os.makedirs(args.outPath, exist_ok=True) # prevents crashes at the end

    # ... THE TASK ...
    execute_task(args,awsHpar)
    inspect_dir(args.outPath,'end-of-job')
    print('M:done')
