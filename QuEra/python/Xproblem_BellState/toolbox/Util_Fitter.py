__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np

#...!...!..................
def fit_data_model(dataP,xBins, oscModel):
    print('FDMshapes',dataP.shape,xBins.shape)
    assert dataP.shape[0]>=2 # expects PEY=2+  for: prob, probErr, [anything else]
    assert dataP.shape[1]==xBins.shape[0]  # the axis must match
    xBins=xBins.astype('float64')
    meas_osc=dataP[0].astype('float64')
    meas_w=1./dataP[1].astype('float64') # Note, lmfit wants 1/err rather than a typical 1/err^2

    print('FDM:inp X:', xBins.shape, xBins.dtype, 'Y:',meas_osc.shape, 'dataP:',dataP.shape)
    assert  xBins.shape== meas_osc.shape
    #print('dump X:',xBins)
    #print('dump Y:',meas_osc)
    #print('dump EY:',dataP[1])
    #print('dump W:',meas_w)
    #meas_w=np.mean(meas_w)
    
    # prime the fitter
    guess = oscModel.guess(meas_osc, x=xBins, verbose=True)# ,period=0.03,ampl=0.5)
    print('\n guess:')
    print(guess)

    ################################################################
    # And now fit the data using the guess as a starting point:
    #meas_w=None
    method='leastsq'  #(Levenberg-Marquardt)
    method='least_squares'  # (Trust Region Reflective)
    #method='L-BFGS-B'  # gives no errors
    #method='powell'  # computes errors, needs: pip install numdifftools
    result = oscModel.fit(meas_osc, params=guess, x=xBins, verbose=True,method=method, weights=meas_w,scale_covar=False)
    # see manual: https://github.com/lmfit/lmfit-py/blob/master/doc/fitting.rst
    
    print('\n fit result:')
    print(result.fit_report() + '\n')
    result.params.pretty_print()

    # ...... explore different methods for error claculation
    # based on https://lmfit.github.io/lmfit-py/confidence.html
    from lmfit import conf_interval,  report_ci
    # run conf_intervale, print report
    sigma_levels = [1, 2, 3]
    ci = conf_interval(result,result, sigmas=sigma_levels)
    print("## Confidence Report:")
    report_ci(ci)   # <-- this is not saved anywhere
    
    ######################################################################
    # Now we'll make some plots of the data and fit.

    #1 guess_osc = oscModel.eval(params=guess, x=xBins)  # to test start only
    fit_osc = oscModel.eval(params=result.params, x=xBins)
    #print('pred:', xBins.shape, xBins.dtype, fit_osc.shape)   
    
    fitMD={}
    fitMD['fit_result']=lmfit2dict(result)
    txt=oscModel.name
    txt=txt.replace('Model(','')
    txt=txt.replace(')','')
    fitMD['fitFuncName']=txt
    return fit_osc,fitMD,result


#...!...!..................
def seed_sin_fit(x,y, verb=1): # find period based on zero-crossings

    # set dumb initialization first  
    xmin = x.min();  xmax = x.max()
    period=(xmax - xmin)/3.
    ampl=0.5
    phase=0
    offset=0.5

    # now do inteligent intialization but quit for unexpected condtions
    
    d5=3 # minimal number of bins which will be used for different steps of this algo
    nbin=x.shape[0]
    #print('SSF: seed_sin_fit nb=',nbin)
    if nbin<20: return [-1 , period, phase, ampl, offset ]
 
    if nbin//d5<4:  return [-2 , period, phase, ampl, offset ]

    assert y.shape[0]==nbin
    yHi=max(y); yLow=min(y); yAvr=y.mean()
    #print('SSF: y range min/max/mid=',yLow,yHi,yAvr)
    offset=yAvr
    
    # decide if start is high--> low or low -> high
    isLow2Hi = y[0] < y[d5*2]
    
    #print('isLow2Hi=',isLow2Hi)
    #assert isLow2Hi

    # count crossing of the median
    iMidL=[];  atRaise=isLow2Hi; i=0; 
    while i<nbin-d5:
        isBelow=y[i]<yAvr
        #print('i',i,y[i],isBelow,atRaise)
        i+=1
        if atRaise and isBelow: continue
        if not atRaise and not isBelow: continue
        iMidL.append(i-1)
        #print('SSF: found mid-cross i=',i,x[i],'atRaise=',atRaise)
        i+=d5
        atRaise=not atRaise
        if len(iMidL) >5: break # no need for more minima

    nCross=len(iMidL)
    if nCross<1:   return [-3 , period, phase, ampl, offset ]
    if nCross==1:  # emergency handling  with 1 zero=-xing
        iavrD=iMidL[0]
    else:  # solid estimate based on 2+ zero crossings
        DL=[iMidL[i]-iMidL[i-1] for i in range(1,nCross) ]
        iavrD=int(np.array(DL).mean())

    if verb>1: # debuging only
        print('SSF: List of mid-crossing K=',nCross,'iavrD=',iavrD,'iMidL=',iMidL)
        for k,i in enumerate(iMidL):
            print(k,i,x[i],y[i])

    if iavrD<d5:   return  [-4 , period, phase, ampl, offset ]

    period=2*(x[iavrD]-x[0])
    phase=-x[iMidL[0]]/period*np.pi*2  # this value is not crucial

    yapog=y[iMidL[0]+iavrD//2] # pick apogeum half-way between crossing
    if isLow2Hi:
        ampl=yapog-yAvr
    else:
        ampl=-yapog+yAvr

    print('seed_sin_fit(): iavrD/bins, period/xunit=',iavrD,period,'yapog,ampl,isLow2Hi=',yapog,ampl,'isL2H:',isLow2Hi,'phase=',phase,'offset=',offset)
   
    return  [0, period, phase, ampl, offset ]





#...!...!..................
def smoothF(x,window_len=20,window='hanning', verb=0):
    """smooth the data using a window with requested size.
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    assert x.ndim == 1
    assert x.size > window_len
    if window_len<3:   return x

    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    if verb: print('smooth Inp',x.shape,x.dtype,'window_len',window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    y=y[(window_len//2-1):-(window_len//2)]
        
    if verb: print('smooth Out',y.shape,y.dtype,window_len//2-1)
    y=y[:x.shape[0]] # hack to get the same out dim for odd window_len
    
    return y




# general utility function for any lmfit result
#...!...!..................
def lmfit2dict(result,covThr=0.50):
    fitD={'largeCorrel':[],'fitPar':{},'fitQA':{}}
    fitD['fitQA']['covThr']=covThr

    fitInfoD={'nfev':'Number of function evaluations', 'success':'fit status','message':'Message about fit','nvarys':'Number of fitted variables','ndata':'Number of data points','nfree':'Degrees of freedom of fit','chisqr':'total chi2','redchi':'chi2/DOF','errorbars':'status'}
    #print('\nfitInfo (manual):')
    for k in fitInfoD.keys():
        val=getattr(result,k)
        print(k,'=',val,'(%s)'%fitInfoD[k],type(val))
        if 'chi' in k: val=float(val)
        if 'errorbars' in k: val='%r'%val

        fitD['fitQA'][k]=[val,fitInfoD[k]]

    bestPar=result.params
    parNameL= result.var_names
    covarOK=result.errorbars
    covarPar=result.covar

    #print('bestPar:',bestPar.keys(),type(bestPar))
    #print('\nparamInfo (manual):',sorted(bestPar),result.nvarys,covThr)

    for i,k in enumerate(parNameL):
        P=bestPar[k]
        val=float(P.value)
        if covarOK:
            err=float(np.sqrt(covarPar[i,i]))
        else:
            err=-1
        #print(i,P.name,val,err,P.vary)
        if val==0.: val=1e-20
        fitD['fitPar'][k]=[val,err,P.vary]
        if covarOK:
            for j in range(i+1,result.nvarys):
                cor=covarPar[j][i]/np.sqrt(covarPar[i][i]*covarPar[j][j])
                #print(cor,i,j,np.abs(cor)>covThr)
                if cor< covThr: continue
                rec=[parNameL[i],parNameL[j],float('%.3f'%cor)]
                fitD['largeCorrel'].append(rec)
    #print('FRDC fitD',fitD)

    #..... Get the names of not varied parameters
    fixed_params = [name for name, param in result.params.items() if not param.vary]
    #print('fixed_params:',fixed_params)
    for pname in fixed_params:
        P=result.params[pname]
        #print('LF2D',pname, P.value,P.vary)
        fitD['fitPar'][pname]=[float(P.value),0,P.vary]
    return fitD



