#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
pulls HW calibration from API and saves it locally as YAML
'''

# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
import numpy as np
from dateutil.parser import parse as date_parse
from datetime import datetime
import pytz
from dateutil.parser import parse as parse_date
import json

sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import access_backend, write_yaml,read_yaml

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm", dest='noXterm', action='store_true',
                         default=False, help="disable X-term for batch mode")
    parser.add_argument("--timeStamp",default='now',help="can provide alternative date as string, e.g. 2019-10-01_21:15_PCT or _UTC")

    parser.add_argument('-b','--backName',default='q20b',
                        choices=['q5o','q14m','q20p','q20b','q20t','q20j','sim'],help="short name for backend" )

    args = parser.parse_args()
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    return args


#...!...!....................
    

#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()

backend=access_backend(args.backName)
args.prjName='hwcal_.%s'%(backend.name())

print('\nmy backend=',backend)
print(backend.status())
print('nqubit=',backend.configuration().n_qubits)

if args.timeStamp=='now':
    mydate=datetime.utcnow().replace(tzinfo=pytz.utc)
else:
    from dateutil import tz
    tzinfos = {"CDT": tz.gettz('US/Central'),'PCT': tz.gettz('US/Pacific')}
    tmStr=args.timeStamp.replace('_',' ')
    print('inp=%s'%tmStr)
    dt3 = parse_date(tmStr, tzinfos=tzinfos)
    print("dt3=",dt3, 'tmz=',dt3.tzinfo)
    mydate=datetime(2019,10,1)
    mydate=dt3

print('set timeStamp=',mydate)

api=backend._api
backend_api=api.client_api.backend(backend.name())
url = backend_api.get_url('properties')
params = {'version': 1}

query = {}
extra_filter = {'last_update_date': {'lt': mydate.isoformat()}}
query['where'] = extra_filter
params['filter'] = json.dumps(query)
response=backend_api.session.get(url, params=params).json()

#Your properties at the timestamps=date
caldtStr=response['last_update_date']
calibT = date_parse(caldtStr)
print('received (get)  calibT=',calibT,'tmz=',calibT.tzinfo,'=%s='%caldtStr)

nowT2=datetime.utcnow().replace(tzinfo=pytz.utc)
delT_h=(nowT2-calibT).total_seconds()/3600.
delT=(nowT2-calibT)
print('\npropertioes updated %s ( %.1f h) ago'%(delT,delT_h),'on:',calibT)

calStr2=calibT.strftime("%Y-%m-%d_%H:%M:%S_%Z")
print('calibT as str w/ tmz=',calStr2)

outF=args.outPath+'calibDB_%s_%s.yaml'%(backend,calStr2)
write_yaml(response,outF)

print('\nskip  USE CASE'); exit(0)
from qiskit import QuantumCircuit, transpile
from qiskit.providers.models import BackendProperties
circF='../noiseStudy/qcdata/grover_3Qas01.qasm'
circOrg=QuantumCircuit.from_qasm_file( circF )
print('circOrg'); print(circOrg)
properties=BackendProperties.from_dict(response)
circOpt=transpile(circOrg,backend=backend, backend_properties=properties,optimization_level=3)
print('circOpt'); print(circOpt)
