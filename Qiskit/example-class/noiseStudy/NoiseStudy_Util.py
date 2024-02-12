import os,sys
from pprint import pprint
import hashlib
import datetime


sys.path.append(os.path.abspath("../../utils/"))
from Circ_Util import  circuit_summary, write_yaml, read_yaml

#...!...!....................
def UUUcirc_2_yaml(name,circ,verb=1):  # packs circuit data for saving
    outD={}
    outD['info']=circuit_summary(name,circ,verb)
    outD['qasm']=circ.qasm()

    hao = hashlib.md5(outD['qasm'].encode())
    outD['md5']=hao.hexdigest()
    #pprint(outD)
    return outD

#...!...!....................
def read_submitInfo(args):
    jid=args.job_id; jid6=args.jid6
    print('verify_jid6 start: job_id=',jid,', jod6=',jid6)
    if jid==None and jid6==None or  jid!=None and jid6!=None:
        print('provide one: jid xor jid6, aborted')
        exit(99)

    # chicken and egg problem
    if jid!=None:
        jid6='j'+jid[-6:]
        print(' will use job_id, proceed')
        
    if jid6!=None:
        assert jid6[0]=='j'
        print(' will use jid6, proceed')

    inpF=args.dataPath+'submInfo-%s.yaml'%jid6
    blob=read_yaml(inpF)
    args.metaD=blob # contains now: submInfo, expInfo
    print('jobInfo: keys=',blob.keys())

    # fill missing info
    if args.job_id==None:  args.job_id=blob['submInfo']['job_id']
    if args.jid6  ==None:  args.jid6=jid6
    return


#...!...!....................
def submitInfo_2_yaml(job_exe, args,circD,noiseModel=None,expMeta=None):  # packs job submission data for saving
    nowT = datetime.datetime.now()
    submInfo={}
    submInfo['job_id']=job_exe.job_id()
    submInfo['jid6']='j'+submInfo['job_id'][-6:] # the last 6 chars in job id , as handy job identiffier
    submInfo['submitDate']=nowT.strftime("%Y-%m-%d %H:%M")
    submInfo['backName']='%s'%job_exe.backend().name()
    submInfo['transpName']=args.transpName
    if expMeta!=None:
        submInfo['num_exp']=len(expMeta)
    
    expInfo={}
    expInfo['noiseModel']=noiseModel
    expInfo['baseCircMD5']=circD['md5']
    expInfo['shots']=args.shots
    expInfo['meas_qubit']=circD['info']['meas_qubit']
    expInfo['expMeta']=expMeta

    outD={'submInfo':submInfo, 'expInfo':expInfo}
    #pprint(outD)
    return outD
  

    
