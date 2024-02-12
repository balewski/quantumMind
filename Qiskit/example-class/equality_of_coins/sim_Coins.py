#!/usr/bin/env python
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
Do coins A and B have the same bias?
based on : Kenneth Rudinger, 1513744.pdf, Sandia NL

'''
# Import general libraries (needed for functions)
import time,os,sys
from pprint import pprint
from scipy import stats
import numpy as np
from dateutil.parser import parse as date_parse

from One_Coin import One_Coin, LogLike,Log2Log

sys.path.append(os.path.abspath("../../utils/"))
from Plotter_Backbone import Plotter_Backbone
from Circ_Util import  npAr_avr_and_err


import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument('-o',"--outPath",default='out',help="output path for plots")
    parser.add_argument( "-X","--no-Xterm",dest='noXterm',action='store_true',
                         default=False, help="disable X-term for batch mode")
 
    parser.add_argument('-P',"--probAB", nargs='+',type=float,default=[0.50,0.50],
                        help="heads probability for coin A and B ")
    parser.add_argument('-s','--shots',type=int,default=8192, help="shots")
    parser.add_argument('-r','--repeatCycle', type=int, default=10000, help="number of copies of the experiment")

    args = parser.parse_args()
    args.prjName='simCoins'
    args.outPath+='/' 
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#............................
#............................
#............................
class Coins_Plotter(Plotter_Backbone):
    def __init__(self, args):
        Plotter_Backbone.__init__(self,args)

#...!...!....................
    def histo_pval(self,pvalL,figId,tit=''):
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,4))

        binsX=np.linspace(-1.5,0,40)
        #binsX=40
        X=np.array(pvalL)
        (avr,std,num,err)=npAr_avr_and_err(X)
        txt2='avr: %.3g +/- %.2g\nstd: %.2g'%(avr,err,std)       
        print(txt2)
        b=np.sum( X<0.1)
        print('pval<0.1  b=%d  frac=%.3f'%(b,b/X.shape[0]))
        #x=np.clip(1.e-2,1.1,x)
        X=X[ X>1e-3]
        #X=X[ X<0.1]
        yA=np.log10(X)
        #yA=x
        print('num data',yA.shape,yA[:20])
        
        #  grid is (yN,xN) - y=0 is at the top,  so dumm
        ax = self.plt.subplot()
        ax.hist(yA,bins=binsX, normed=True)

        ax.text(0.15,0.8,txt2,color='b',transform=ax.transAxes)
        ax.set(title='coin H0'+tit, xlabel='log10(pvalue)', ylabel='fract. exper./bin')
        ax.set_yscale('log')
        ax.set_ylim(0.05,)

        ax.grid()



#=================================
#=================================
#  M A I N
#=================================
#=================================
args=get_parser()
plot=Coins_Plotter(args)

coinL=[]
for iC, prob in enumerate(args.probAB):    
    coin=One_Coin('coin%d'%(iC),prob)
    coinL.append(coin)

print('\n Experiment repeat',args.repeatCycle)

pvalL=[]
for i in range(args.repeatCycle):    
    for coin in coinL: coin.run(args.shots)
    score=Log2Log(coinL[0].data, coinL[1].data)
    pval=stats.chi2.cdf(score,1.)
    #print('pp',pval)
    if i<10:
        print('score=',score,', p-value=%.3g'%pval)
    if i%1000==0: print('do i=%d, p-value=%.3g'%(i,pval))
    pvalL.append(pval)

tit=',truth: Pa=%.3f  pB=%.3f, shots=%d, nExp=%d'%(args.probAB[0],args.probAB[1],args.shots, args.repeatCycle)    
plot.histo_pval(pvalL,10,tit)
plot.display_all('ana', tight=True)
    
