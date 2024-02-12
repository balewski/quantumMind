#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 Mathematical solution to the MIS problem
reads NAME.quera.h5  as input
saves h5 in ana/NAME.math.h5
Graphics optional
'''

import os
from pprint import pprint
import numpy as np
import json 
import networkx as nx  # for graph generation
from bitstring import BitArray


from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.PlotterQuEra import PlotterQuEra

from ProblemMaxIndependentSet import ProblemMIS

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3,4],  help="increase output verbosity", default=1, dest='verb')
         
    parser.add_argument("--basePath",default='env',help="head dir for set of experiments, or env--> $QuEra_dataVault")

    parser.add_argument('-e',"--expName",  default='exp_62a15b',help='AWS-QuEra experiment name assigned during submission')

    # plotting
    parser.add_argument("-p", "--showPlots",  default='g r d ', nargs='+',help="abc-string listing shown plots")
    parser.add_argument( "-X","--noXterm", dest='noXterm',  action='store_true', default=False, help="disable X-term for batch mode")
   
    args = parser.parse_args()
    # make arguments  more flexible
    if 'env'==args.basePath: args.basePath= os.environ['QuEra_dataVault']
    args.dataPath=os.path.join(args.basePath,'jobs')
    args.outPath=os.path.join(args.basePath,'ana')
    args.showPlots=''.join(args.showPlots)
    
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!....................
def _draw_MIS_graph(G,misId,pos,nAtom,ax):
    A=BitArray(uint=misId,length=nAtom)
    bitstr=A.bin  # needed for nodes coloring        
    vert_colors = ['red' if (bitc == '1') else 'aqua' for bitc in bitstr]
    nx.draw(G, pos = pos, ax=ax, with_labels=True, node_color=vert_colors)

#............................
#............................
#............................
class Plotter(PlotterQuEra):
    def __init__(self, args):
        PlotterQuEra.__init__(self,args)
        
    #...!...!....................
    def MIS_sample(self,task,card,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(9,11))

        pd=task.meta['payload']
        nAtom= pd['num_atom_in_clust']
        pos=task.atoms_pos_um()[:nAtom]
        G=task.graph

        if card=='max':
            misV=task.expD['true_MISs_max_card']
            gCard=task.meta['true_math']['max_card']
            
        if card=='min':
            misV=task.expD['true_MISs_min_card']
            gCard=task.meta['true_math']['min_card']
            
        print('PL: %d MIS exist for card=%d'%( misV.shape[0],gCard))
        nG=min(nrow*ncol, misV.shape[0])
        for k in range(nG):
           ax = self.plt.subplot(nrow,ncol,1+k)
           ax.set_aspect(1.)
           misId=int(misV[k])
           
           tit='MIS id=%d card=%d'%(misId,gCard)
           print('sol:',tit)
           ax.set(title=tit)
           _draw_MIS_graph(G,misId,pos,nAtom,ax)
           txt1=None
           if k==0: txt1=pd['info'].replace(",","\n")
           if txt1!=None: ax.text(0.02,0.89,txt1,color='g', transform=ax.transAxes,va='top')
       
        
        
    #...!...!....................
    def graphs(self,task,figId=3):
        figId=self.smart_append(figId)
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,5))
        
        pd=task.meta['payload']
        nAtom= pd['num_atom_in_clust']
        pos=task.atoms_pos_um()[:nAtom]
        
        for k in range(2):
           ax = self.plt.subplot(nrow,ncol,1+k)
           ax.set_aspect(1.)
           if k==0:
               G=task.graph               
               tit='atoms graph: %d nodes'%G.number_of_nodes()
               vert_colors='aqua'
               
           if k==1:
               G=task.complGraph               
               tit='complement graph: %d edges'%G.number_of_edges()
               vert_colors='y'
               
           #print('sol:',tit)
           ax.set(title=tit)

           nx.draw(G, pos = pos, ax=ax, with_labels=True, node_color=vert_colors)

           txt1=None
           if k==0: txt1=pd['info'].replace(",","\n")
           if txt1!=None: ax.text(0.02,0.89,txt1,color='g', transform=ax.transAxes,va='top')
           
            

        
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=4)
                    
    inpF=args.expName+'.quera.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.dataPath,inpF))
         
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            for x in  expD:
                if 'qasm3' in x : continue
                print(x,expD[x])  
        stop3

        
    task= ProblemMIS(args,expMD,expD)

    task.mathematicalSolution()
    
    #...... WRITE  OUTPUT
    outF=os.path.join(args.outPath,expMD['short_name']+'.math.h5')
    write4_data_hdf5(task.expD,outF,expMD)

    # ----  just plotting
    args.prjName=expMD['short_name']
    plot=Plotter(args)

    #expMD['timeLR']=[0.,4.1]  # (ns)  time range clip
    if expMD['payload']['num_clust']==1:
        expMD['XrangeLR']=[-5.,25.]  # (um)  x-range range clip
        expMD['YrangeLR']=[-5.,25.]  # (um)  x-range range clip
    
    if 'r' in args.showPlots:
        plot.show_register(task, what_to_draw="circle")
         
    if 'd' in args.showPlots:  # drives
        plot.global_drive(task)
        
    if 'g' in args.showPlots: # graphs
        plot.graphs( task )
        
    if 'm' in args.showPlots: # MIS for max  cardinality
        plot.MIS_sample( task, card='max' )
        
    if 'n' in args.showPlots: # MIS for min  cardinality
        plot.MIS_sample( task, card='min' )
        
    plot.display_all(png=1)

    print('M:done')


