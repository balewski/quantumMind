__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import time
import ruamel.yaml  as yaml
import networkx as nx
import json


#...!...!..................
def read_yaml(ymlFn, verb=1):
        print('  read  yaml:',ymlFn,end='')
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)
        ymlFd.close()
        print(' done, size=%d'%len(bulk))
        if verb>1: print('see keys:',list(bulk.keys()))
        elif verb>2: pprint(bulk)
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
        start = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
                print('  closed  YAML:',ymlFn,' size=%.1f kB'%xx,'  elaT=%.1f sec'%(time.time() - start))


#...!...!..................
def graph_to_JSON(G):
    graph_data = nx.node_link_data(G)
    tmpGR = json.dumps(graph_data)
    return tmpGR

#...!...!..................
def graph_from_JSON(strJ):
    #print('aa',strJ)
    # Load the JSON string into a dictionary
    graph_data = json.loads(strJ)
    #print('graph_data=',type(graph_data)); print(graph_data)  
    G = nx.node_link_graph(graph_data)
    return G

        
''' - - - - - - - - -
time-zone aware time, usage:

*) get current date:
t1=time.localtime()   <type 'time.struct_time'>

*) convert to string:
timeStr=dateT2Str(t1)

*) revert to struct_time
t2=dateStr2T(timeStr)

*) compute difference in sec:
t3=time.localtime()
delT=time.mktime(t3) - time.mktime(t1)
delSec=delT.total_seconds()
or delT is already in seconds
'''

#...!...!..................
def dateT2Str(xT):  # --> string
    nowStr=time.strftime("%Y%m%d_%H%M%S_%Z",xT)
    return nowStr

#...!...!..................
def dateStr2T(xS):  #  --> datetime
    yT = time.strptime(xS,"%Y%m%d_%H%M%S_%Z")
    return yT

   
