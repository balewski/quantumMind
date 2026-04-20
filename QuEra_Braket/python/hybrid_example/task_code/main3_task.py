#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os,sys
import json
from pprint import pprint
sys.path.append('customer_code/extracted/task_code/') # because AWS is running Code As Subprocess
from toolbox.Util1 import util_func1,env_2_dict,dump_file
from toolbox.Util2 import util_func2

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase output verbosity", default=1)
    parser.add_argument("--num_iter", default=3, help="number of iterations")
    parser.add_argument("--device_arn", default=None, help="Amazon Resource Name")
    parser.add_argument("--backend_short", default=None, help="short backend name")
    parser.add_argument("--short_name", default=None, help="short job  name")

    args = parser.parse_args()
    # transformations improving flexibility
    if type(args.num_iter)==type(str()): args.num_iter=int(args.num_iter)  # AWS passes only strings
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!....................
def inspect_dir(head,text='my123'):
    print('\n----Inspect %s  head=%s'%(text,head))   
    directory_contents = os.listdir(head)
    # Print the list of contents
    for item in directory_contents:
        c='/' if os.path.isdir(os.path.join(head,item)) else ''        
        print(item+c)

#...!...!....................
def traverse_AWS_dirs(path0):
    path=path0
    path+='/customer_code'
    inspect_dir(path, 'customer_code dir')
    path+='/extracted'
    inspect_dir(path, 'extracted dir')
    path+='/task_code'
    inspect_dir(path, 'task_code dir')
    path+='/toolbox'
    inspect_dir(path, 'toolbox dir')
    envD=env_2_dict('SM_TRAINING_ENV',verb=0)

    inspect_dir('/opt/ml/input', 'inp1 dir')
    inspect_dir('/opt/ml/input/data', 'inp2 dir')
    inspect_dir('/opt/ml/input/data/input', 'inp3 dir')

    #.... save config as JSON
    hpar=envD['hyperparameters']
    print('\nAWS hpar:'); pprint(hpar)
    outF='/opt/ml/output/data/ahs_task.conf.json'
    with open(outF, 'w') as json_file:
        md={'name1':'aaa123'}
        md['hyperparameters']= hpar
        json.dump(md, json_file)
    print('RHS: saved '+outF)

    pathOut=envD['output_data_dir']
    inspect_dir(pathOut,'pathOut')

    pathInp=envD['channel_input_dirs']['input']
    inspect_dir(pathInp,'pathinp')


#...!...!....................
def execute_task(isHybrid=True):
    print('I AM execute_task(), isHybrid:',isHybrid)
    args=get_parser()
    
    print('START hybrid task, num_iter=%d  device_arn=%s backN=%s'%(args.num_iter,args.device_arn,args.backend_short))
    # ... inspect directories ....
    path=os.getcwd()
    inspect_dir(path, 'working dir')

    if isHybrid:
        traverse_AWS_dirs(path)
        if 0:  dump_file('braket_container.py')
   
    util_func1()
    util_func2()
 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == "__main__":
    print('I AM MAIN()')
    
    execute_task(isHybrid=False)

    print('M:done')
